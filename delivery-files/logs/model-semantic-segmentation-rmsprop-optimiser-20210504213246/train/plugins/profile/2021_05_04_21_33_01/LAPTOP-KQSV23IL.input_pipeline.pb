	?t???@?t???@!?t???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?t???@~?D?8z?@1<iᲲ;@Af/?N??Iq?Ws? 4@rEagerKernelExecute 0*	a??"??[@2U
Iterator::Model::ParallelMapV2 հ????!4???<VB@) հ????14???<VB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??3????!?[???4@)cAaP?є?1u*?x??2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ʼUס??!?ي:}X7@)ձJ??^??1?[~G0@:Preprocessing2F
Iterator::ModelU/??dƫ?!&{???XH@)?h9?Cm??1??<?
(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceD2??z???!?????@)D2??z???1?????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?T??C??!ڄAx?I@)v?1<??x?1y?j???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?j??g?!????Q>@)?j??g?1????Q>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaph"lxz???!
?̖~9@)??z`?1Z??@??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??"??X@Qi	?P???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~?D?8z?@~?D?8z?@!~?D?8z?@      ??!       "	<iᲲ;@<iᲲ;@!<iᲲ;@*      ??!       2	f/?N??f/?N??!f/?N??:	q?Ws? 4@q?Ws? 4@!q?Ws? 4@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??"??X@yi	?P???