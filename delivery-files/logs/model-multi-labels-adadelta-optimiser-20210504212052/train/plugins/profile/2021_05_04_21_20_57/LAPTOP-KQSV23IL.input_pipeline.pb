	???̪?@???̪?@!???̪?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???̪?@??\??]?@1R,??
*@A<L??????IW?o"@rEagerKernelExecute 0*	??Q??Z@2U
Iterator::Model::ParallelMapV2&z?????!?&??2:@)&z?????1?&??2:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat8fٓ????!Z_#:@)ra?r??1?8sβ7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?kBZcЙ?!???̒X7@)q??sC??1j?w?k1@:Preprocessing2F
Iterator::Model??6???!Sf>??jE@)?&??d??1?VX??0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice!???3z?!/?{Wm?@)!???3z?1/?{Wm?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+?m?????!???{b?L@)uۈ'?y?117??^E@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor^????k?!?JLad?@)^????k?1?JLad?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??˚X???!?c??69@)\?J?`?1??<????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?M????X@Q?Y?;:??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??\??]?@??\??]?@!??\??]?@      ??!       "	R,??
*@R,??
*@!R,??
*@*      ??!       2	<L??????<L??????!<L??????:	W?o"@W?o"@!W?o"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?M????X@y?Y?;:??