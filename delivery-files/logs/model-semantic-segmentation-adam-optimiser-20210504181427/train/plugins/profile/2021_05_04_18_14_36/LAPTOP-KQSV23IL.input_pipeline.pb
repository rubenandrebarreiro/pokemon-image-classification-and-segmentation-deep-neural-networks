	̷>?§@̷>?§@!̷>?§@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC̷>?§@?S????@1b??!i6@A*??ѫ??I???f] @rEagerKernelExecute 0*	?Q???X@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceS?!?uq??!P ?ad1;@)S?!?uq??1P ?ad1;@:Preprocessing2U
Iterator::Model::ParallelMapV2??8ӄ???!??̗??7@)??8ӄ???1??̗??7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??p??|??!??$^O3@)???s????16z?/Z0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?s]?@??!X??>?E@)???W:??1??58??-@:Preprocessing2F
Iterator::Model?(ϼv??!??t#?,?@)??X?_"~?1?0?.?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???V_]??!B?"w?4Q@)46<?Rv?1??m??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??^?2?g?!?<W=s?@)??^?2?g?1?<W=s?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?-???1??!?dE?'?E@)?A?p?-^?1~){t???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??????X@QÖ???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?S????@?S????@!?S????@      ??!       "	b??!i6@b??!i6@!b??!i6@*      ??!       2	*??ѫ??*??ѫ??!*??ѫ??:	???f] @???f] @!???f] @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??????X@yÖ???