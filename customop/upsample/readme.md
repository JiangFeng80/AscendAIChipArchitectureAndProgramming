
```python
def darknet_upsample_cce(
	shape,
	dtype, 
	size, 
	data_format="channels_last",	
	kernel_name="cce_darknet_upsample", 
	need_build=False, 
	need_print=False
)
```

**Description**

Upsample operator in Darknet.


**Args:**

- shape: input tensor's shape (4D or 5D)
- dtype: input tensor's dtype, support:`float16,float32,int32,int8,uint8`
- size: the scale of each axis (4D or 5D)
- data_format: one of `channels_last` (default) or `channels_first`. For example(4D): The ordering of the dimensions in the inputs.`channels_last` corresponds to inputs with shape
     `(batch, height, width, channels)` while `channels_first` corresponds to inputs with shape `(batch, channels, height, width)`.
- kernel_name: op's kernel func name, optional
- need_build: whether build CCEC kernel, default is `False`, optional
- need_print: whether print IR, default is `False`, optional

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`
