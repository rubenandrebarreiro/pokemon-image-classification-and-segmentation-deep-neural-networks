	.rO李作.rO李作!.rO李作      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC.rO李作U?-家誧@1a?喲`6@AEH楱噙?I?5峖#?@rEagerKernelExecute 0*R膆???Y@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?K?^I??!?媴	{?A@)愒???1思|硤"@@:Preprocessing2U
Iterator::Model::ParallelMapV2}?!8.???!:>?? ?5@)}?!8.???1:>?? ?5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateJ'L5???!.娃/?5@)?u?X???1+???T0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipw;S頛??!1HS皦@)??輩x???1/??;j?*@:Preprocessing2F
Iterator::Model憱F"4???!9??釵?<@)?1??|z?1 ???Z@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice恇J抭?v?!?? ??@)恇J抭?v?1?? ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorI?2?嶢?!2?o錉?@)I?2?嶢?12?o錉?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapE硤?甸??!?家7@)?/K;5?[?1XH???i??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?4鋦N埁@Q????嚶??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	U?-家誧@U?-家誧@!U?-家誧@      ??!       "	a?喲`6@a?喲`6@!a?喲`6@*      ??!       2	EH楱噙?EH楱噙?!EH楱噙?:	?5峖#?@?5峖#?@!?5峖#?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?4鋦N埁@y????嚶??