?	@?0`a
?@@?0`a
?@!@?0`a
?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC@?0`a
?@I?Qݡ@1q???X-@A?]i????I?d?z??@rEagerKernelExecute 0*	sh??|Z@2U
Iterator::Model::ParallelMapV2M?D?u???!A3'u?!<@)M?D?u???1A3'u?!<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??)X?l??!??ҟ?8@)??OVW??1?@?G?5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??j?=&??!?&???A@)?w??ۏ?1???-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatez?΅?^??!?g?l?3@)???????1]
???)@:Preprocessing2F
Iterator::Modelē???G??!<}X?B@)x?캷"??1n?w? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?p?;z?!׉?'d?@)?p?;z?1׉?'d?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}]??t??!????`?O@)o/i??Qu?1 h?)?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????h?!C?z??&@)?????h?1C?z??&@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI.??T?X@Q??h)?U??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	I?Qݡ@I?Qݡ@!I?Qݡ@      ??!       "	q???X-@q???X-@!q???X-@*      ??!       2	?]i?????]i????!?]i????:	?d?z??@?d?z??@!?d?z??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q.??T?X@y??h)?U???
"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???z??!???z??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!?&y?O??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??a?l??!L??T???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput|?kKo??!??q????0"]
;pokemon-images-multi-label-classification/activation_6/Relu_FusedConv2D^?%,?G??!?z??0??"]
;pokemon-images-multi-label-classification/activation_7/Relu_FusedConv2DM]E?B??!++泦@??"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterf?r????!????GS??0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInputWP&?c
??!??V???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput???????!?J? &???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_4/Conv2D/Conv2DBackpropInputConv2DBackpropInputl~??B???!o?N:\??0Q      Y@Y??o??@a??/?W@q?????W@y?s	?G???"?

both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 