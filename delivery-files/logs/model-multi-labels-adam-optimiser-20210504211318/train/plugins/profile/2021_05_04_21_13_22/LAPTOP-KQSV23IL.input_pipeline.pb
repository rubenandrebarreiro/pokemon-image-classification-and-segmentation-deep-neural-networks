	u?)??@u?)??@!u?)??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCu?)??@X??"'՘@1??+f?(@AfO?sp??I%̴?+[@rEagerKernelExecute 0*	?v???`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatЗ??\4??!?=$?J=@)??P?v0??1&`?^:@:Preprocessing2U
Iterator::Model::ParallelMapV2??,??Ρ?!?L)???9@)??,??Ρ?1?L)???9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??<e5??!??Z?<?>@)?~m?????1*?]?Y/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{??{?ʔ?!ru)?$.@){??{?ʔ?1ru)?$.@:Preprocessing2F
Iterator::ModelD?.l?V??!?? ?f1@@)ѕT? ??1
b?BH@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip^??a?Q??!9???L?P@)???!??1?g5*?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF?n?1p?!?d!a_@)F?n?1p?1?d!a_@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??҈?}??!??YU?M@@)ro~?D?d?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI`?"?X@Q}?ϱnw??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	X??"'՘@X??"'՘@!X??"'՘@      ??!       "	??+f?(@??+f?(@!??+f?(@*      ??!       2	fO?sp??fO?sp??!fO?sp??:	%̴?+[@%̴?+[@!%̴?+[@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`?"?X@y}?ϱnw??