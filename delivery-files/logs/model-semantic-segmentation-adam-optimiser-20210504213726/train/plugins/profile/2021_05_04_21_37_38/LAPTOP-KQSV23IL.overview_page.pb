?	e?,?=*?@e?,?=*?@!e?,?=*?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCe?,?=*?@qX?-
?@1KW??x*9@AE?*k????I	???c>@rEagerKernelExecute 0*	???Mb?X@2U
Iterator::Model::ParallelMapV2?b?0???!c#???C@)?b?0???1c#???C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???iOə?!@  ??V9@)H?9??*??1I???%?5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate#?GG???!?????K5@)?V?I???1J_????.@:Preprocessing2F
Iterator::Model??????!"????PG@)ҧU??f~?1?ŋ?w?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?b?=yx?!&?@?@)?b?=yx?1&?@?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipcz?(??!?c3
h?J@)?x?s?1????>?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorh??n?l?!?????t@)h??n?l?1?????t@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?%z????!? ??7@)?J?E?]?1?'2M0??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI~k????X@Qp??,	??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	qX?-
?@qX?-
?@!qX?-
?@      ??!       "	KW??x*9@KW??x*9@!KW??x*9@*      ??!       2	E?*k????E?*k????!E?*k????:		???c>@	???c>@!	???c>@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q~k????X@yp??,	???
"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?;駛??!?;駛??0"?
kgradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter?&?????!F?v?Ms??0"?
igradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter??2{Um??!a??*????0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!Ma.?yO??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter4,p??*??!ZlJ?7Z??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter[????!?1?Hq%??0"?
kgradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_6/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter??F?
Z??!H??????0"R
4pokemon-images-semantic-segmentation/conv2d_8/Conv2DConv2D???&$??!??o?B???0"y
[gradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose/conv2d_transpose/Conv2DConv2D??(?QĎ?!?k2?????0"{
]gradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_1/conv2d_transpose/Conv2DConv2D?|?+???!?5^????0Q      Y@YxN[@a?K??W@q@"??76X@yDV_Ө???"?

both?Your program is POTENTIALLY input-bound because 99.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 