?	??\??0@??\??0@!??\??0@	?J?R+????J?R+???!?J?R+???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??\??0@??V???@1Ԟ?sbW%@A?c*?ߗ?I?7?k?g@Y[??vN???rEagerKernelExecute 0*	??x?&y[@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat? 3??O??!p????B@)??]Pߢ?1?¶W?@@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?&??0???!???<?`4@)?&??0???1???<?`4@:Preprocessing2U
Iterator::Model::ParallelMapV2???]M???!?/<??0@)???]M???1?/<??0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??.5B???!n?QۙA@)?t?? ??1????T?+@:Preprocessing2F
Iterator::Model?b.???!?8?/?7@)?k$	???1?e$?z?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{i? ?w??!?q?S@)?:??Kt?1jQ%	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??5"g?!]W??v@)??5"g?1]W??v@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap~?????!v8???A@)?0{?v?Z?1???|????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?16.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?J?R+???I~??!?lA@Q.????O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??V???@??V???@!??V???@      ??!       "	Ԟ?sbW%@Ԟ?sbW%@!Ԟ?sbW%@*      ??!       2	?c*?ߗ??c*?ߗ?!?c*?ߗ?:	?7?k?g@?7?k?g@!?7?k?g@B      ??!       J	[??vN???[??vN???![??vN???R      ??!       Z	[??vN???[??vN???![??vN???b      ??!       JGPUY?J?R+???b q~??!?lA@y.????O@?
"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterҿL????!ҿL????0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterD???C???!??}?}???0"L
%Adam/Adam/update_16/ResourceApplyAdamResourceApplyAdamO+??`??!_?1{a??"]
;pokemon-images-multi-class-classification/activation_6/Relu_FusedConv2D7?P?L???!m?Ŭ???"]
;pokemon-images-multi-class-classification/activation_7/Relu_FusedConv2D?}?i????!?*?#'[??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter4o??? ??!??*?C??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput?j}?o3??!+?:?????0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput??R????!??w'???0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter0Ӌ??#??!!|?????0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter9vgM???!??&????0Q      Y@YVg?{?*@a5s???U@q1*?s%?R@y?dU?????"?
both?Your program is POTENTIALLY input-bound because 17.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?16.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?74.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 