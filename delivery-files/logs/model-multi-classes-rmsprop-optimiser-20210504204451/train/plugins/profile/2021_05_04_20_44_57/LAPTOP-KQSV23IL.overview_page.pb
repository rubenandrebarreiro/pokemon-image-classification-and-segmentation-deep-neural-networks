?	?????2?@?????2?@!?????2?@	;N?0
Վ?;N?0
Վ?!;N?0
Վ?"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?????2?@@7
?@1: 	?vr,@AKVE?ɨ??I>????@Y)??5??rEagerKernelExecute 0*	M7?A`5^@2U
Iterator::Model::ParallelMapV2y??.??!\+d=@)y??.??1\+d=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???
??!*jB?>@)|~!<ڠ?1?ri?f=;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????ם?!fB C8@)??Xl????1b???72@:Preprocessing2F
Iterator::Model?e3????!????LB@)?n/i?ց?1?_??;?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?? ?X4}?!$???E?@)?? ?X4}?1$???E?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip[닄????!q^?T?O@)h?.?KRy?1L?Yv?v@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorc?#?w~q?!?????F@)c?#?w~q?1?????F@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA?m??!?q ??;@)?$??p?1?J??W#@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9;N?0
Վ?I܋wO??X@Qev???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	@7
?@@7
?@!@7
?@      ??!       "	: 	?vr,@: 	?vr,@!: 	?vr,@*      ??!       2	KVE?ɨ??KVE?ɨ??!KVE?ɨ??:	>????@>????@!>????@B      ??!       J	)??5??)??5??!)??5??R      ??!       Z	)??5??)??5??!)??5??b      ??!       JGPUY;N?0
Վ?b q܋wO??X@yev????
"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??@\??!??@\??0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??AE??!4ө?B9??0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???ӡ?!"?=0g???0"]
;pokemon-images-multi-class-classification/activation_6/Relu_FusedConv2D???? ̡?!?ȟRg??"]
;pokemon-images-multi-class-classification/activation_7/Relu_FusedConv2D?JQ ???!??X?h??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterh???F??!??)Y????0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput&?4?/??!Cs?oH???0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput?????Ȝ?!?<?H?|??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?\?]4??!???0??0"]
;pokemon-images-multi-class-classification/activation_2/Relu_FusedConv2D?N??l??!v??x:???Q      Y@Y<Eg@(@a?K??{mW@qQ?:?ÕS@y?ڃ???"?

both?Your program is POTENTIALLY input-bound because 99.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?78.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 