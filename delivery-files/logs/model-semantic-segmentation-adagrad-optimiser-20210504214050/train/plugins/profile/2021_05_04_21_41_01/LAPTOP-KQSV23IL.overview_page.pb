?	?z?F??@?z?F??@!?z?F??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?z?F??@?ʢ????@1SB??^B9@A?? ?=	@I?M~?N?@rEagerKernelExecute 0*	w??/?W@2U
Iterator::Model::ParallelMapV2i?V?Θ?!
2*?	?9@)i?V?Θ?1
2*?	?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateS$_	?Ğ?!s?7????@)????~???1?Һ?:?9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?.?.ǘ?!$???%?9@)S#?3????1?V?B\?6@:Preprocessing2F
Iterator::ModelЛ?T[??!Y6?@@)???H???1????h @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceO??唀x?!????n@)O??唀x?1????n@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???/JЯ?!??$???P@)????kw?1?&"y?O@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor1]??ah?!5ѬiLN	@)1]??ah?15ѬiLN	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$}ZE??!??:O?A@)??\5?a?1???yw|@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?q??X@Q?}?U????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ʢ????@?ʢ????@!?ʢ????@      ??!       "	SB??^B9@SB??^B9@!SB??^B9@*      ??!       2	?? ?=	@?? ?=	@!?? ?=	@:	?M~?N?@?M~?N?@!?M~?N?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?q??X@y?}?U?????
"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter
????!
????0"?
kgradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter|z?/??!??ۅ???0"?
igradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter)???6???!|??Pǲ?0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterp2?*|Ֆ?!?Hu??|??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter)??r??! )m-̀??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?g?????!C-?5??0"?
kgradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_6/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterߚ???C??!]V?u???0"R
4pokemon-images-semantic-segmentation/conv2d_8/Conv2DConv2DL??C????!H??l???0"y
[gradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose/conv2d_transpose/Conv2DConv2Dd????#??!????P???0"{
]gradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_1/conv2d_transpose/Conv2DConv2D??1???!??ɛc???0Q      Y@Y?\?PۭA@a?ѹW)P@qa??kX@y٣>妀?"?

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
Refer to the TF2 Profiler FAQb?96.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 