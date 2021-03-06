?	kb?????@kb?????@!kb?????@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCkb?????@????G?@1?E&?ט'@AI,)w????I??|zl?@rEagerKernelExecute 0*	?A`??
[@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??P?n??!??q)?x?@))A?G???1?+??=?;@:Preprocessing2U
Iterator::Model::ParallelMapV2"R?.???!tf6(7@)"R?.???1tf6(7@:Preprocessing2F
Iterator::Model??Pj/???!??Ga??C@)Y?O0???1?b?x??/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate)??5??!]????5@)䠄????1`??O??,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceA,?9$???!?t5]?*@)A,?9$???1?t5]?*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?{H??߰?!
t??xN@)???5x?1%Ԋ???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor)	????q?!r"v??$@))	????q?1r"v??$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap˟o???!?B?ԓ 8@)?dc?1?+??(@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?k?(?X@Q??J?k??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????G?@????G?@!????G?@      ??!       "	?E&?ט'@?E&?ט'@!?E&?ט'@*      ??!       2	I,)w????I,)w????!I,)w????:	??|zl?@??|zl?@!??|zl?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?k?(?X@y??J?k???
"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!??????0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter6??8'ٰ?!????z???0"\
0Adagrad/Adagrad/update_16/ResourceApplyAdagradV2ResourceApplyAdagradV2?E4????!?Z%q???"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter&ZP.?I??!???0???0"]
;pokemon-images-multi-class-classification/activation_6/Relu_FusedConv2D8?P??"??!????H???"]
;pokemon-images-multi-class-classification/activation_7/Relu_FusedConv2D?U
? ??!???z_J??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter0???J???!?)YϨ???0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput?nq{?v??!iW?>???0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput(U???X??!B???6??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput ?+?@]??!.?_?LB??0Q      Y@Yo0E>?+@a?Y7?"?U@q???2??W@ye???Ɖ??"?

both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 