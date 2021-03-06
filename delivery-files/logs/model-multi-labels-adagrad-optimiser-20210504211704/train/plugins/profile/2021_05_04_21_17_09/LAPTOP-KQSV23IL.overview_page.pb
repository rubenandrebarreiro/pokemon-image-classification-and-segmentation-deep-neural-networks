?	_{fy??@_{fy??@!_{fy??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC_{fy??@j???,r?@1??b??(@A?ui???Im 6 B?@rEagerKernelExecute 0*	?????V@2U
Iterator::Model::ParallelMapV2?v??/??!??y?C+?@)?v??/??1??y?C+?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?j-?B;??!?ZB??8@)?o?^}<??1✱?ٜ5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?X?U???!G???29@)\?~l???1?2?x?I2@:Preprocessing2F
Iterator::Model????1v??!A?h???C@)5?+-#?~?1??? ? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?6???Z??!?'?gXHN@)???B?i~?1u?W??= @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice*?~???y?!k2????@)*?~???y?1k2????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorg??j+?g?!.?ce?	@)g??j+?g?1.?ce?	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????ߙ?!n??;@)s?m?B<b?1?????y@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI? ?{??X@Qv?#B<??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	j???,r?@j???,r?@!j???,r?@      ??!       "	??b??(@??b??(@!??b??(@*      ??!       2	?ui????ui???!?ui???:	m 6 B?@m 6 B?@!m 6 B?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q? ?{??X@yv?#B<???
"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter3Վk9??!3Վk9??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterV??M<??!Dl?ë??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterB?G??!P?!?g???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput5P? ??!]???/p??0"\
0Adagrad/Adagrad/update_16/ResourceApplyAdagradV2ResourceApplyAdagradV22??????!՟n?????"]
;pokemon-images-multi-label-classification/activation_7/Relu_FusedConv2DZ?j?/ؤ?!????????"]
;pokemon-images-multi-label-classification/activation_6/Relu_FusedConv2Di?m֧ʤ?!M???5#??"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Ń?????!
1??t???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput-??????!04,֔???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput&{???v??!?cBfo???0Q      Y@Y???
b@@a?????P@q???o:?W@y{?養??"?

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
Refer to the TF2 Profiler FAQb?95.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 