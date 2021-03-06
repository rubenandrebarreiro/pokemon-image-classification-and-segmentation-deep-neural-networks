?	?|???@?|???@!?|???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?|???@M1A?r?@1????Y,@A??V?????I?Y.?@rEagerKernelExecute 0*	g??|?_@2U
Iterator::Model::ParallelMapV2??
~b??!?I??@@)??
~b??1?I??@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatt}???!????b}6@)?l??<+??1\q???3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatePō[̟?!??y???8@)Fx{???1Ec?f?<1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??<?!7??!???~p/N@)7+1?J??1???h??$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceu???????!b?#??@)u???????1b?#??@:Preprocessing2F
Iterator::Model??7?-:??!Wy???C@)"??I`??1J]??p@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǄ?K??k?!?"??|?@)Ǆ?K??k?1?"??|?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap&6׆???!Ѽ?5%?;@)]???Ej?1'?2??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??Hf]?X@Q$@??LQ??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	M1A?r?@M1A?r?@!M1A?r?@      ??!       "	????Y,@????Y,@!????Y,@*      ??!       2	??V???????V?????!??V?????:	?Y.?@?Y.?@!?Y.?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??Hf]?X@y$@??LQ???
"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?#??S??!?#??S??0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterJ#?j???!|&##???0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??5?ۡ?!?	_8???0"]
;pokemon-images-multi-class-classification/activation_6/Relu_FusedConv2D??!r?֡?!?k??????"]
;pokemon-images-multi-class-classification/activation_7/Relu_FusedConv2D6^???С?!~??#o??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?o4?uD??!~R`A???0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput?M????!???B???0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput6???GÜ?!1?̎v{??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput+????H??!t)??0??0"]
;pokemon-images-multi-class-classification/activation_2/Relu_FusedConv2D?s)?>??!?`\?????Q      Y@Y<Eg@(@a?K??{mW@qr??q9 X@yVAv?O???"?

both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 