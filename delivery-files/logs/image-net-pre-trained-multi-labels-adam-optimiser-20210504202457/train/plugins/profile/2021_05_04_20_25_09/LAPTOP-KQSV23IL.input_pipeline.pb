	it??R?@it??R?@!it??R?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCit??R?@?Z??@1K??F>QK@AD?l???@I??O?m?@rEagerKernelExecute 0*	??SK$?RA2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorK?r ?@!.=$??X@)K?r ?@1.=$??X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetchd??A??!?:??;?)d??A??1?:??;?:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelismτ&?%??!??Z??K?)??d9	??1????;?:Preprocessing2Y
"Iterator::Model::PrivateThreadPool?J???!b5?5A}P?)	???W?1A??C??$?:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap?W? ?@!@?V??X@)?9???z?1٫??v?!?:Preprocessing2F
Iterator::Model?e?ت?!?!??T?Q?)?"??l?14Ğ#;??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?Dt?ʼX@QR???L???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Z??@?Z??@!?Z??@      ??!       "	K??F>QK@K??F>QK@!K??F>QK@*      ??!       2	D?l???@D?l???@!D?l???@:	??O?m?@??O?m?@!??O?m?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Dt?ʼX@yR???L???