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

#include <Python.h>
#include "proto/caffe/caffe.pb.h"
#include "custom/custom_op.h"
#include "framework/omg/register.h"
#include "framework/omg/omg_types.h"
#include "operator.h"
#include "attr_value.h"
#include <memory>
#include <string>
#include <vector>

using namespace ge;
namespace domi
{
// Caffe ParseParams
Status ParseParams_Upsample(const Message* op_origin, ge::Operator& op_dest)
{
    // trans op_src to op_dest
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

    if (nullptr == layer)
    {
        printf("Dynamic cast op_src to LayerParameter failed.\n");
        return FAILED;
    }
    //TODO: Please add the UpsampleParameter in caffe.proto, you can name it your own way
    const caffe::UpsampleParameter& param = layer->upsample_param();

    if(param.has_scale())
    {
        op_dest.SetAttr("scale", AttrValue::CreateFrom<AttrValue::INT>(param.scale()));
    }

    return SUCCESS;
}
// #### Obtains the processing function of the output tensor description. 
Status TFInferShapeAndType_Upsample(const ge::Operator& op, vector<ge::TensorDesc>& v_output_desc)
{
    v_output_desc.push_back(op.GetInputDesc(0));
    auto tensorDesc      = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();

    std::string data_format;
    int64_t scale = 1;

    ge::AttrValue scaleAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("scale", scaleAttrValue)) || 
        (ge::GRAPH_SUCCESS != scaleAttrValue.GetValue<AttrValue::INT>(scale)))
    {
        printf("GetOpAttr scale failed!\n");
    }

    shape.SetDim(2, shape.GetDim(2)*scale);
    shape.SetDim(3, shape.GetDim(3)*scale);

    v_output_desc[0].SetShape(shape);
    
    return SUCCESS;
}


// build Te Binary file
Status TFBuildTeBin_Upsample(const ge::Operator& op, TEBinInfo& te_bin_info)
{
    std::string FilePath   = "";
    std::string FuncName   = "";
    std::string KernelName = "";
    std::string data_format = "";
    int64_t scale = 1;
    auto tensorDesc = op.GetInputDesc(0);
    auto shape      = tensorDesc.GetShape();

    ge::AttrValue scaleAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("scale", scaleAttrValue)) || 
        (ge::GRAPH_SUCCESS != scaleAttrValue.GetValue<AttrValue::INT>(scale)))
    {
        printf("GetOpAttr scale failed!\n");
    }
    //TODO: the path below should be the path darknet_upsample.py, please modify the path
    FilePath   = "../python/darknet_upsample.py";
    FuncName   = "darknet_upsample_cce";

    // check input tensor shape whether equal to 2
    if (shape.GetDimNum() != 4)
    {
        printf("The shape size is %d, which is not 4!", (int32_t)shape.GetDimNum());
        return FAILED;
    }
    
    KernelName = "darknet_upsample_" + 
        std::to_string(shape.GetDim(0)) + "_" + 
        std::to_string(shape.GetDim(1)/16) + "_" + 
        std::to_string(shape.GetDim(2)) + "_" + 
        std::to_string(shape.GetDim(3));
    
    // i => int; s => string; f => dobule; O => bool, and bool value is Py_True or Py_False
    te::BuildTeCustomOp(te_bin_info.ddk_version, op.GetName(), FilePath, FuncName,
                    "(i,i,i,i,i), s, (i,i), s, s, O",
                    shape.GetDim(0), shape.GetDim(1)/16, shape.GetDim(2), shape.GetDim(3), 16,
                    "float16",
                    scale, scale,
                    data_format.c_str(),
                    KernelName.c_str(),
                    Py_True);

    // set bin and json path info
    te_bin_info.bin_file_path  = "./kernel_meta/" + KernelName + ".o";
    te_bin_info.json_file_path = "./kernel_meta/" + KernelName + ".json";

    return SUCCESS;
}

// test_reduction is the type name of the operator in the OM model. 
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive. 
REGISTER_CUSTOM_OP("upsample") 
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("Upsample")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_Upsample)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .InferShapeAndTypeFn(TFInferShapeAndType_Upsample)       // Set output description and datatype function
    .TEBinBuildFn(TFBuildTeBin_Upsample) // Build Te op binary function
    .ImplyType(ImplyType::TVM) // Implementation type. Enumerated type, The options are as follows: TVM, AI_CPU.
    .Formats({DOMI_TENSOR_NC1HWC0},{DOMI_TENSOR_NC1HWC0});

}  // namespace domi
