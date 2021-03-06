?	#f?y??@#f?y??@!#f?y??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC#f?y??@?蜟???@1???Ĭ?9@A????I;?G?!@rEagerKernelExecute 0*	(1?*Y@2U
Iterator::Model::ParallelMapV2?q6??!o?^mR+;@)?q6??1o?^mR+;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?0a4+ۗ?!m;e?%7@)7n1?7??1???3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateAc&Q/??! ??v7@)?sCSv???1?W?5q1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Y??/-??!?sPl?A@)??CV??1`?SS?'@:Preprocessing2F
Iterator::Model?}"O??!@???[?A@)N?&?O:??1#?? ʶ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip'h??'???!?*? RP@)?HP?x?1!?>?=@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?4Lk?x?!
b&v?@)?4Lk?x?1
b&v?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:̗`m?!d??A@):̗`m?1d??A@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIćH;?X@Q?;x????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?蜟???@?蜟???@!?蜟???@      ??!       "	???Ĭ?9@???Ĭ?9@!???Ĭ?9@*      ??!       2	????????!????:	;?G?!@;?G?!@!;?G?!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qćH;?X@y?;x?????
"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???P?U??!???P?U??0"?
kgradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter??,?ɗ?!BS?????0"?
igradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter???z??!?L??I??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?D???G??!?]2ڭ۷?0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter_P?2ؕ??!۱??#???0"?
kgradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_6/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterm???%H??!???????0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?lA5??!57$Y?+??0"R
4pokemon-images-semantic-segmentation/conv2d_8/Conv2DConv2D???????!?H?:?`??0"y
[gradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose/conv2d_transpose/Conv2DConv2D??g?-͍?!??O?=??0"{
]gradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_1/conv2d_transpose/Conv2DConv2D???hMl??!????j??0Q      Y@Y?@&?d@a???C??W@q?9lAX@yr?%y???"?

both?Your program is POTENTIALLY input-bound because 99.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 