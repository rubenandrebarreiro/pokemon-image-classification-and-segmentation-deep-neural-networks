	??9???@??9???@!??9???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??9???@9?Z?9|?@1?8?? 6L@A???V?/??I?={.Ss@rEagerKernelExecute 0*	?n??JK@2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetchw.???v??!????F%L@)w.???v??1????F%L@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelismЛ?T[??!図???U@))	??????1?\[;??>@:Preprocessing2F
Iterator::Model??o???!      Y@)T8?T?m?1;	?+?l@:Preprocessing2Y
"Iterator::Model::PrivateThreadPool???????!m?A?1YW@)M?n?k?1n8?o?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIґ?o?X@Qux?????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	9?Z?9|?@9?Z?9|?@!9?Z?9|?@      ??!       "	?8?? 6L@?8?? 6L@!?8?? 6L@*      ??!       2	???V?/?????V?/??!???V?/??:	?={.Ss@?={.Ss@!?={.Ss@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qґ?o?X@yux?????