?	?w???@?w???@!?w???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?w???@A??_?Ȟ@1Ӥt{))@A??L????IP?}:3@rEagerKernelExecute 0*	??C??]@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???cw???!???/=*>@)Tƿϸp??1????]?:@:Preprocessing2U
Iterator::Model::ParallelMapV2?? ?=??!??<?$?8@)?? ?=??1??<?$?8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate4??????!R?&?e8@)?????%??1.3???2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipZ??? ʹ?!?K???P@)?3?9A???1?7oH?P'@:Preprocessing2F
Iterator::ModelR?y9쾣?!?h$?@@)???cw???1???/=*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??????y?!???3-@)??????y?1???3-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?,'???p?!=??t??
@)?,'???p?1=??t??
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?O?Y????!;?!???9@)????Wb?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIy/????X@QGCh?????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	A??_?Ȟ@A??_?Ȟ@!A??_?Ȟ@      ??!       "	Ӥt{))@Ӥt{))@!Ӥt{))@*      ??!       2	??L??????L????!??L????:	P?}:3@P?}:3@!P?}:3@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qy/????X@yGCh??????
"\
1Adadelta/Adadelta/update_16/ResourceApplyAdadeltaResourceApplyAdadelta?<}7??!?<}7??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?4?? -??!F?b?2??0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterOj'????!n]?ی8??0"]
;pokemon-images-multi-class-classification/activation_6/Relu_FusedConv2DJ?2????!?Wx?v`??"]
;pokemon-images-multi-class-classification/activation_7/Relu_FusedConv2D??P?8l??!qKfp½??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterF???"??!3T\?B??0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???А??!I1?4???0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInputGuF?.4??!??G?????0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput(?H?*??!?1?
???0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?кeo}??!????????0Q      Y@Y????`?E@aSb?1L@q05?=??W@yx?ϣ?Y??"?

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
Refer to the TF2 Profiler FAQb?95.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 