	4???eѲ@4???eѲ@!4???eѲ@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC4???eѲ@[Υ?V??@1?ui?pK@A-"??`??I????=@rEagerKernelExecute 0*	U㥛?PD@2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch?ajK???!??D?NJ@)?ajK???1??D?NJ@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism???N??!͢w̤?T@)???|?r??1?oި"?>@:Preprocessing2Y
"Iterator::Model::PrivateThreadPoolUi?k|&??!??&{?W@)մ?i?{m?1??yu-?!@:Preprocessing2F
Iterator::ModelTUh ?ͤ?!      Y@)???M?qj?1???MX?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIU8?X@Qd???1:??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	[Υ?V??@[Υ?V??@![Υ?V??@      ??!       "	?ui?pK@?ui?pK@!?ui?pK@*      ??!       2	-"??`??-"??`??!-"??`??:	????=@????=@!????=@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qU8?X@yd???1:??