/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "hiaiengine/ai_model_manager.h"
#include <unistd.h>
using namespace std;
using namespace hiai;

/*
*** brief:Reads input data
*/
char* ReadBinFile(const char *file_name, uint32_t *fileSize)
{
    std::filebuf *pbuf;
    std::ifstream filestr;
    size_t size;
    filestr.open(file_name, std::ios::binary);
    if (!filestr)
    {
        return nullptr;
    }

    pbuf = filestr.rdbuf();
    size = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
    pbuf->pubseekpos(0, std::ios::in);
    if (size <= 0)
    {
	return nullptr;
    }
    char * buffer = (char*)malloc(size);
    if (nullptr == buffer)
    {
        return nullptr;
    }

    pbuf->sgetn(buffer, size);
    *fileSize = size;

    filestr.close();
    return buffer;
}

class SampleModelListener: public IAIListener
{
public:
    virtual void OnProcessDone(const AIContext &context, int result, const std::vector<std::shared_ptr<IAITensor>> &model_output) override
    {
        isFinished = true;
        for(uint32_t i = 0; i< model_output.size(); i++)
        {
            shared_ptr<AINeuralNetworkBuffer> temp_tensor = static_pointer_cast<AINeuralNetworkBuffer>(model_output[i]);
            printf("output[%d].name :%s\n", i, temp_tensor->GetName().c_str());
            printf("output[%d].size :%d\n", i, temp_tensor->GetSize());

            float * result = (float *)temp_tensor->GetBuffer();
            for (uint32_t j = 0; j < (uint32_t)(temp_tensor->GetSize() / 4); j++)
            {
                printf("index[%d]:%f\n", j, *(result + j));
            }
         }
        printf("Predict done.taskid=%s", context.GetPara("taskid").c_str());
    }
public:
    bool isFinished = false;
};

int main(int argc, char* argv[])
{
    printf("bbit main start.\n");

    if (argc != 3)
    {
        cout << "usage: " << argv[0] << " modelfile  datafile" << endl;
        return -1;
    }

    // If the listener is set, the model reasoning is asynchronous execution. If the listener is not set, it is invoked synchronously.
    shared_ptr<hiai::IAIListener> listener(new SampleModelListener);
    string key = "";

    // Do not set the listener.
    AIModelManager model_mngr;
    AIModelDescription model_desc;
    AIConfig config;
    AIContext context;
    context.AddPara("taskid", "001");
    string MODEL_PATH = argv[1];
    model_desc.set_path(MODEL_PATH.c_str());
    model_desc.set_type(0);
    vector<AIModelDescription> model_descs;
    model_descs.push_back(model_desc);
    AIStatus ret = model_mngr.Init(config, model_descs);
    model_mngr.SetListener(listener);
    // input tensor
    // input tensor will be reset after the image data is read. The function is only used for initialization.
    AITensorDescription input_tensor_desc = AINeuralNetworkBuffer::GetDescription();
    shared_ptr<IAITensor> input_tensor = AITensorFactory::GetInstance()->CreateTensor(input_tensor_desc);
    shared_ptr<AISimpleTensor> input_simple_tensor = static_pointer_cast<AISimpleTensor>(input_tensor);
    if (nullptr == input_tensor)
    {
        printf("nullptr == input_tensor.\n");
        return -1;
    }
    // Reads image data
    uint32_t image_data_size = 0;

    std::string IMAGE_FILE_PATH = argv[2];

    float* image_data = (float*)ReadBinFile(IMAGE_FILE_PATH.c_str(), &image_data_size);
    if (nullptr == image_data)
    {
        printf("ReadBinFile failed bin file path= %s \n", IMAGE_FILE_PATH.c_str());
        return -1;
    }

    // Set the pointer and length of the picture data address to input_simple_tensor.
    // Set this parameter to true. The tensor releases the image_data address after the life cycle ends.
    // The subscriber does not need to release the tensor address.
    input_simple_tensor->SetBuffer((void*)image_data, image_data_size,true);
    printf("SetBuffer ok.\n");

    // Sets the model input and output.
    vector<shared_ptr<IAITensor>> model_input;
    vector<shared_ptr<IAITensor>> model_output;
    model_input.push_back(input_tensor);
    // The timeout parameter is invalid temporarily. The model manager does not support the timeout mechanism.
    uint32_t timeout = 0;

    if(model_mngr.IsPreAllocateOutputMem())
    {
        ret = model_mngr.CreateOutputTensor(model_input, model_output);
        printf("CreateOutputTensor, ret = %d\n", ret);
    }

    // Starting Model Reasoning
    printf("Start process.\n");
    ret = model_mngr.Process(context, model_input, model_output, 0);
    if (ret != hiai::SUCCESS)
    {
        printf("model_mngr Process failed!\n");
        return -1;
    }

    printf("wait process.\n");
    shared_ptr<SampleModelListener> model_listener = static_pointer_cast<SampleModelListener>(listener);
    while(!model_listener->isFinished)
    {
        sleep(100);
    }

    return 0;
}
