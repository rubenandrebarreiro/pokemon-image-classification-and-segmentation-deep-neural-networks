	_{fy??@_{fy??@!_{fy??@      ??!       "?
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
	j???,r?@j???,r?@!j???,r?@      ??!       "	??b??(@??b??(@!??b??(@*      ??!       2	?ui????ui???!?ui???:	m 6 B?@m 6 B?@!m 6 B?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q? ?{??X@yv?#B<??