?	D?!T?1@D?!T?1@!D?!T?1@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCD?!T?1@??5(@1~r 
V%@A???w????I>x?҆@rEagerKernelExecute 0*X9?ȦV@)      =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatef`X???!ax?????@)iV?y˕?1?sɲ?}7@:Preprocessing2U
Iterator::Model::ParallelMapV2??:?f???!ߜ)?c3@)??:?f???1ߜ)?c3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeats?蜟???!?????2B@)wMHk:??1nZ+84?2@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoroe??2???!?hқ??1@)oe??2???1?hқ??1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?>tA}?|?!9ص	@)?>tA}?|?19ص	@:Preprocessing2F
Iterator::Model\?????!?w???9@)d??uy?1??5?q@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\Z?{,??!?>b??R@)hY????p?1Ǫ[U9@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/?e?????!??a??@@)?'eRC[?1?RA?eb??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?16.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIs???^?B@Q?7?8O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??5(@??5(@!??5(@      ??!       "	~r 
V%@~r 
V%@!~r 
V%@*      ??!       2	???w???????w????!???w????:	>x?҆@>x?҆@!>x?҆@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qs???^?B@y?7?8O@?
"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterJpSݼ??!JpSݼ??0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??x?C??!??fP???0"L
%Adam/Adam/update_16/ResourceApplyAdamResourceApplyAdam?s?W$??!@hief	??"]
;pokemon-images-multi-class-classification/activation_6/Relu_FusedConv2D=J?;￧?!ϺV4b???"]
;pokemon-images-multi-class-classification/activation_7/Relu_FusedConv2DW??ѫ???!2N^?O??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?i&?;.??!hc???0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput	gi????!IH0f?x??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput6w??&??!0??@????0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltervr?
K??!?.????0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Fwa??!?9??'???0Q      Y@YVg?{?*@a5s???U@qa?!??V@y???|Ğ?"?
both?Your program is POTENTIALLY input-bound because 20.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?16.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?91.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 