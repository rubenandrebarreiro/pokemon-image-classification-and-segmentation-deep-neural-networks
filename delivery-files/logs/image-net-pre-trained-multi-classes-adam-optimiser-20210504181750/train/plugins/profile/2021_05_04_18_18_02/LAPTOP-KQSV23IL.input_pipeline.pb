	??);=?@??);=?@!??);=?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??);=?@???K?@1??/??H@Az?m?(??I???0?@rEagerKernelExecute 0*	!?rh??E@2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetcheo)狽??!Oy?ݝ?J@)eo)狽??1Oy?ݝ?J@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism?H? O??!??3q??T@)?C??????1$??	
=@:Preprocessing2Y
"Iterator::Model::PrivateThreadPool??ڦx\??!?p<g[?V@)R<??kp?1??C?O~"@:Preprocessing2F
Iterator::Modelg?+??2??!      Y@)??Քdm?1?{?$? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIO?՗?X@Q^,?
Z???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???K?@???K?@!???K?@      ??!       "	??/??H@??/??H@!??/??H@*      ??!       2	z?m?(??z?m?(??!z?m?(??:	???0?@???0?@!???0?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qO?՗?X@y^,?
Z???