	8??E?@8??E?@!8??E?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC8??E?@Ѱue?@1???Ŋr(@A??S??IAJ?i@rEagerKernelExecute 0*	5^?I`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatC?B?Y???!???O?>@)M?d??7??1??SpB:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate&?(??=??!?B<??>@)???P?v??1?[?j?9@:Preprocessing2U
Iterator::Model::ParallelMapV2?j?TQ??!??6	?5@)?j?TQ??1??6	?5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????U??!??r???Q@)?W歺??1??럾 @:Preprocessing2F
Iterator::Model????????!0?5:??<@)I???????185`??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??[?d8~?!??6Es@)??[?d8~?1??6Es@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????<,t?!??3??@)????<,t?1??3??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????????!)^ ???@@)g|_\??f?1???O_g@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?:??p?X@Q???????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ѱue?@Ѱue?@!Ѱue?@      ??!       "	???Ŋr(@???Ŋr(@!???Ŋr(@*      ??!       2	??S????S??!??S??:	AJ?i@AJ?i@!AJ?i@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?:??p?X@y???????