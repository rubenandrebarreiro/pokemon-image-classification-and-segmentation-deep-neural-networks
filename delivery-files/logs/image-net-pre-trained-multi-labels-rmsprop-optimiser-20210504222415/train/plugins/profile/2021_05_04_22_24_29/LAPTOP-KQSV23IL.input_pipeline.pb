	s?????@s?????@!s?????@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCs?????@??@Ȣ?@14?y?SeN@A$?6?De??IA?3*@rEagerKernelExecute 0*	??CsYdVA2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???c???@!>?^???X@)???c???@1>?^???X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch????&M??!%???ZE?)????&M??1%???ZE?:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism]????۩?!?ܹ?1L?)?jGq?:??1???S??,?:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMapg)YN???@!]F????X@)~b??Um?1??#??:Preprocessing2F
Iterator::Model?Hg`?e??!!???P?)????U?l?1?It????:Preprocessing2Y
"Iterator::Model::PrivateThreadPool?)?????!j???wN?)?)??F?k?1Q?" 1.?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?/x???X@Q???ޒ??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??@Ȣ?@??@Ȣ?@!??@Ȣ?@      ??!       "	4?y?SeN@4?y?SeN@!4?y?SeN@*      ??!       2	$?6?De??$?6?De??!$?6?De??:	A?3*@A?3*@!A?3*@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?/x???X@y???ޒ??