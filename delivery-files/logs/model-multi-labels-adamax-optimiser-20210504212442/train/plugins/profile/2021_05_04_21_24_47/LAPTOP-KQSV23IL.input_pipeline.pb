	=C8f???@=C8f???@!=C8f???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC=C8f???@?W?????@1t]?@?)@Ad?&???I??l;m?@rEagerKernelExecute 0*	أp=
X@2U
Iterator::Model::ParallelMapV2ut\????!???q[;@)ut\????1???q[;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?aL?{)??!?M?y?<@)?ZH???1???9&9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatej?drjg??!?u@$??8@)I?s
????1??`??1@:Preprocessing2F
Iterator::Model?2?FY???!ED??qB@)??#bJ$??1T??j!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????z?!?k?G<@)????z?1?k?G<@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\??.?u??!??hl??O@)d!:?z?1??iE?o@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??d?VA??!ӭ?7??<@)??c?n?1??1?HN@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???Ik?!?@? ?@)???Ik?1?@? ?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?5<???X@Q?6???-??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?W?????@?W?????@!?W?????@      ??!       "	t]?@?)@t]?@?)@!t]?@?)@*      ??!       2	d?&???d?&???!d?&???:	??l;m?@??l;m?@!??l;m?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?5<???X@y?6???-??