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

// #### Obtains the processing function of the output tensor description. 
Status TFReductionInferShapeAndType(const ge::Operator& op, vector<ge::TensorDesc>& v_output_desc)
{
    auto tensorDesc      = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();
    int64_t axis = -1;

    ge::AttrValue axisAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("axis", axisAttrValue)) || (ge::GRAPH_SUCCESS != axisAttrValue.GetValue<AttrValue::INT>(axis)))
    {
        printf("Get axis failed!\n");
    }

    if (axis < 0) axis += shape.GetDimNum();

    if (axis < 0 || axis >= shape.GetDimNum())
    {
        printf("invalid axis:%d, dim_size:%d\n", (int32_t)axis, (int32_t)shape.GetDimNum());
        return PARAM_INVALID;
    }
    shape.SetDim(axis, 1);
    tensorDesc.SetShape(shape);
    v_output_desc.push_back(tensorDesc);

    return SUCCESS;

}


// build Te Binary file
Status TFReductionBuildTeBin(const ge::Operator& op, TEBinInfo& te_bin_info)
{
    std::string FilePath   = "";
    std::string FuncName   = "";
    std::string KernelName = "";
    std::string operation  = "";
    int64_t     axis       = -1;
    float       coeff      = 1;
    // ### Parses the operation parameter. 
    ge::AttrValue operationAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("operation", operationAttrValue)) || (ge::GRAPH_SUCCESS != operationAttrValue.GetValue<AttrValue::STR>(operation)))
    {
        // ### Add exception handling and maintenance informatio
        printf("GetOpAttr operation failed!\n");
    }

    // ### Parse the axis parameter. 
    ge::AttrValue axisAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("axis", axisAttrValue)) || (ge::GRAPH_SUCCESS != axisAttrValue.GetValue<AttrValue::INT>(axis)))
    {
        printf("GetOpAttr axis failed!\n");
    }

    // ### Parse the coeff parameter. 
    ge::AttrValue coeffAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("coeff", coeffAttrValue)) || (ge::GRAPH_SUCCESS != coeffAttrValue.GetValue<AttrValue::FLOAT>(coeff)))
    {
        printf("GetOpAttr coeff failed!\n");
    }
    // ### Parse input tensor description 
    TensorDesc input_desc      = op.GetInputDesc(0);

    // ### Parse the input shape value and check whether the value is 2
    if(input_desc.GetShape().GetDimNum() != 2)
    {
        printf("The shape size is %d, which is not 2!", (int32_t)input_desc.GetShape().GetDimNum());
        return FAILED;
    }
    FilePath   = "topi/cce/caffe_reduction_layer";
    FuncName   = "caffe_reduction_layer_cce";
    KernelName = "cce_reductionLayer_1_10_1_1_float16__3_SUMSQ_1_0";

    // i => int; s => string; f => dobule; O => bool, and bool value is Py_True or Py_False
    te::BuildTeCustomOp(te_bin_info.ddk_version, op.GetName(), FilePath, FuncName,
                    "(i,i,i,i), s, i, s, f, s", input_desc.GetShape().GetDim(0), input_desc.GetShape().GetDim(1),
                    1, 1, "float16", axis, operation.c_str(), coeff,
                    KernelName.c_str());

    // set te op json to te_bin_info 
    te_bin_info.bin_file_path  = "./kernel_meta/" + KernelName + ".o";
    te_bin_info.json_file_path = "./kernel_meta/" + KernelName + ".json";

    return SUCCESS;
}

REGISTER_CUSTOM_OP("custom_reduction") //test_reduction is the type name of the operator in the OM model. It can be specified randomly and cannot be the same as an existing type name. It is case sensitive. 
    .FrameworkType(TENSORFLOW)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("Reduction")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(AutoMappingFn)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .InferShapeAndTypeFn(TFReductionInferShapeAndType)       // Set output description and datatype function
    .TEBinBuildFn(TFReductionBuildTeBin) // Build Te op binary function
    .ImplyType(ImplyType::TVM) // Implementation type. Enumerated type, The options are as follows: TVM, AI_CPU.
    .Formats({DOMI_TENSOR_ND},{DOMI_TENSOR_ND});   //  #### Format of the input and output

}  // namespace domi
