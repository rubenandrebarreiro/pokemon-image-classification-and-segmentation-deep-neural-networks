	N?0?IR?@N?0?IR?@!N?0?IR?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCN?0?IR?@???dyH?@1???8???@Ay?&1???I???;?f@rEagerKernelExecute 0*	3333yA2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?f׽??f@!??5?X@)?f׽??f@1??5?X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch'?O:?`??!^???l&??)'?O:?`??1^???l&??:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism?>????!oZ???)????䀍?1??6'qM??:Preprocessing2F
Iterator::Model??T?????!?#?P??)&?R?o*r?1Q??Nd?:Preprocessing2Y
"Iterator::Model::PrivateThreadPool{3j?J>??!>?8攘?)?Go???j?1Ꜳ(t?]?:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap?p;4??f@!&??N?X@)͓k
dvf?1??<??X?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIZ?hv?fV@Q.U?L?$@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???dyH?@???dyH?@!???dyH?@      ??!       "	???8???@???8???@!???8???@*      ??!       2	y?&1???y?&1???!y?&1???:	???;?f@???;?f@!???;?f@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qZ?hv?fV@y.U?L?$@