?	?S?[t?@?S?[t?@!?S?[t?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?S?[t?@A??Հɗ@1???;?(@A?&??0??IN??}??@rEagerKernelExecute 0*	X9??vvd@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatex
?Rς??! ??Dv>M@)ly?z?L??1*1???iI@:Preprocessing2U
Iterator::Model::ParallelMapV2?=?4a??!U??UU0@)?=?4a??1U??UU0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?1k?M??!r?,???+@)?W?}W??1???E(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea???????!?_ç[?@)a???????1?_ç[?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?߽?Ƅ??!3??W?S@)R<??k??1L?0vy?@:Preprocessing2F
Iterator::Modelv??????!?3/d?*5@)????*4??1?z*+U@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??p???g?!x?oHD??)??p???g?1x?oHD??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$??t?(??!?!??\N@)s?<G??d?1??p?ռ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??M???X@Qٻ???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	A??Հɗ@A??Հɗ@!A??Հɗ@      ??!       "	???;?(@???;?(@!???;?(@*      ??!       2	?&??0???&??0??!?&??0??:	N??}??@N??}??@!N??}??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??M???X@yٻ????
"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter<#??s??!<#??s??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??,f+z??!???????0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter|????k??!@v???,??0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInputouy?d??!?ӕ????0"L
%Adam/Adam/update_16/ResourceApplyAdamResourceApplyAdamT?}??!?+΀?R??"]
;pokemon-images-multi-label-classification/activation_7/Relu_FusedConv2D???-???! O7????"]
;pokemon-images-multi-label-classification/activation_6/Relu_FusedConv2D??H??z??!jH,/u??"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterr??????!j?%?????0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInputrL?D????!?&?+???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput߂ x????!T7?:v??0Q      Y@Y?m۶m?&@aI?$I?$V@q
?????W@yɚ{2????"?

both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 