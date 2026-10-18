# Search-G1 A/B committed-progress comparison

This report uses already-read small records only. It neither changes training nor proves final acceptance.
Periodic actual-EM-priority selected mixing is reported separately from ordinary pre-label selection.
Probe classification metrics are not QA EM/F1. Different actor snapshots produce different diagnostic labels;
across-outer metrics are not a fixed-label learning curve. Missing records remain pending.

| Arm | Actual complete | Included records | Ordinary from6 | Mean selected mixed | Mean whole wall |
|---|---:|---:|---:|---:|---:|
|A|35|35|27|53.8701%|783.633399s|
|B|35|35|27|49.3996%|791.787133s|

## Scheduled QA source aggregates

These are source-reported QA aggregates, not a substitute for final question-identity and worker-output acceptance.

| Arm | Outer | Question count | EM | F1 | Actual worker exits |
|---|---:|---:|---:|---:|---|
|A|5|512|27.5391%|36.1807%|{"0": 0, "1": 0, "2": 0, "3": 0}|
|A|50|pending|pending|pending|pending|
|A|100|pending|pending|pending|pending|
|A|150|pending|pending|pending|pending|
|A|200|pending|pending|pending|pending|
|B|5|512|28.5156%|37.3150%|{"0": 0, "1": 0, "2": 0, "3": 0}|
|B|50|pending|pending|pending|pending|
|B|100|pending|pending|pending|pending|
|B|150|pending|pending|pending|pending|
|B|200|pending|pending|pending|pending|

## A: per-outer selected outcomes

Uncontinued candidate groups have no labels; selected metrics do not establish candidate-pool accuracy.

| Outer | Periodic actual-EM priority | Selected mixed | All wrong | All correct | Predicted mean mixing | Whole wall seconds |
|---|---|---:|---:|---:|---:|---:|
|1|False|77.3438%|14.4531%|8.2031%|91.2262%|589.547167|
|2|False|76.5625%|12.3047%|11.1328%|91.9372%|974.194057|
|3|False|74.8047%|14.2578%|10.9375%|91.7354%|784.152546|
|4|False|76.9531%|10.9375%|12.1094%|91.4645%|748.519070|
|5|False|79.6875%|10.7422%|9.5703%|91.1364%|899.092150|
|6|False|78.9062%|8.0078%|13.0859%|93.3254%|720.920595|
|7|False|79.1016%|6.4453%|14.4531%|93.2178%|787.663080|
|8|False|71.8750%|7.2266%|20.8984%|93.2291%|806.225901|
|9|False|71.4844%|5.6641%|22.8516%|93.2217%|779.165029|
|10|True|100.0000%|0.0000%|0.0000%|86.6929%|1085.731906|
|11|False|67.7734%|5.0781%|27.1484%|93.2324%|776.805157|
|12|False|68.9453%|5.4688%|25.5859%|93.1906%|766.868309|
|13|False|64.4531%|7.0312%|28.5156%|93.2839%|787.143075|
|14|False|61.1328%|7.8125%|31.0547%|93.1976%|776.777266|
|15|False|58.5938%|7.0312%|34.3750%|93.0880%|851.985211|
|16|False|63.6719%|6.2500%|30.0781%|93.2426%|821.133276|
|17|False|53.3203%|6.0547%|40.6250%|93.2474%|770.842922|
|18|False|58.0078%|4.2969%|37.6953%|93.2325%|760.657223|
|19|False|58.2031%|4.2969%|37.5000%|93.1554%|791.527423|
|20|True|100.0000%|0.0000%|0.0000%|80.4640%|1073.248644|
|21|False|51.5625%|6.6406%|41.7969%|93.2456%|783.814007|
|22|False|50.0000%|6.4453%|43.5547%|93.2994%|778.304099|
|23|False|50.3906%|6.2500%|43.3594%|93.1877%|790.699770|
|24|False|47.0703%|8.2031%|44.7266%|93.2119%|770.669612|
|25|False|46.0938%|7.2266%|46.6797%|93.1838%|794.013374|
|26|False|42.1875%|7.6172%|50.1953%|93.2136%|793.924603|
|27|False|42.9688%|9.7656%|47.2656%|93.1989%|781.229297|
|28|False|45.1172%|5.6641%|49.2188%|93.2163%|783.170393|
|29|False|40.6250%|8.3984%|50.9766%|93.1536%|784.348091|
|30|True|100.0000%|0.0000%|0.0000%|68.7217%|1053.015451|
|31|False|39.0625%|6.6406%|54.2969%|93.1778%|760.187401|
|32|False|38.8672%|4.1016%|57.0312%|93.2674%|776.860021|
|33|False|36.5234%|5.2734%|58.2031%|93.1435%|785.522531|
|34|False|35.9375%|8.5938%|55.4688%|93.2100%|788.145843|
|35|False|32.6172%|8.0078%|59.3750%|93.2003%|789.498260|

## A: periodic same-data probe comparisons

Train is the original unresampled distribution. Each pre/post pair uses the same trajectories and labels.

| Outer | Head / distribution | BA before→after | Ordinary accuracy before→after | Trajectory BCE before→after | Mixed BCE before→after |
|---|---|---:|---:|---:|---:|
|10|frozen / train|74.1672%→74.1672%|76.3281%→76.3281%|0.429435→0.429435|0.571557→0.571557|
|10|frozen / test|56.8660%→56.8660%|59.5703%→59.5703%|0.750121→0.750121|0.800626→0.800626|
|20|frozen / train|66.4423%→66.4423%|68.6719%→68.6719%|0.486028→0.486028|0.704767→0.704767|
|20|frozen / test|50.6534%→50.6534%|54.8828%→54.8828%|0.847845→0.847845|0.868183→0.868183|
|30|frozen / train|64.2235%→64.2235%|66.4844%→66.4844%|0.501670→0.501670|0.743420→0.743420|
|30|frozen / test|51.3894%→51.3894%|55.8594%→55.8594%|0.930569→0.930569|0.848909→0.848909|

Actual stored global probe loss rows: 0.
QA source records, full paired metrics, recall/confusion matrices, balancing statistics and clocks are retained in the JSON companion.

## B: per-outer selected outcomes

Uncontinued candidate groups have no labels; selected metrics do not establish candidate-pool accuracy.

| Outer | Periodic actual-EM priority | Selected mixed | All wrong | All correct | Predicted mean mixing | Whole wall seconds |
|---|---|---:|---:|---:|---:|---:|
|1|False|78.5156%|12.3047%|9.1797%|91.2240%|612.038294|
|2|False|76.5625%|12.3047%|11.1328%|91.5987%|976.122527|
|3|False|76.5625%|10.5469%|12.8906%|90.9172%|814.823989|
|4|False|79.8828%|9.3750%|10.7422%|89.6553%|738.386350|
|5|False|74.2188%|8.3984%|17.3828%|87.5429%|898.212757|
|6|False|70.5078%|6.4453%|23.0469%|91.7834%|718.561613|
|7|False|64.8438%|6.8359%|28.3203%|91.2852%|785.688747|
|8|False|65.6250%|4.4922%|29.8828%|90.4367%|806.788981|
|9|False|56.8359%|4.2969%|38.8672%|89.7191%|785.590051|
|10|True|100.0000%|0.0000%|0.0000%|66.0736%|1074.769756|
|11|False|59.9609%|3.5156%|36.5234%|91.5355%|795.213678|
|12|False|57.8125%|5.4688%|36.7188%|92.6263%|802.280774|
|13|False|58.3984%|4.6875%|36.9141%|92.9872%|774.691658|
|14|False|61.5234%|5.4688%|33.0078%|93.1832%|791.022987|
|15|False|54.2969%|6.4453%|39.2578%|93.1715%|806.515953|
|16|False|57.2266%|9.5703%|33.2031%|93.2739%|799.042085|
|17|False|57.8125%|8.2031%|33.9844%|93.2079%|836.576202|
|18|False|55.8594%|10.3516%|33.7891%|93.0796%|779.442195|
|19|False|53.9062%|11.1328%|34.9609%|92.9434%|825.930922|
|20|True|100.0000%|0.0000%|0.0000%|81.9799%|1077.304980|
|21|False|50.0000%|19.9219%|30.0781%|92.6169%|809.343775|
|22|False|48.8281%|26.5625%|24.6094%|92.6232%|792.286245|
|23|False|44.9219%|26.7578%|28.3203%|92.5945%|789.380796|
|24|False|47.6562%|30.6641%|21.6797%|92.5996%|783.550526|
|25|False|42.1875%|36.5234%|21.2891%|92.7843%|792.702408|
|26|False|41.9922%|45.7031%|12.3047%|92.7154%|787.918655|
|27|False|32.2266%|57.4219%|10.3516%|92.7869%|779.375658|
|28|False|31.6406%|61.1328%|7.2266%|92.9917%|781.771250|
|29|False|26.1719%|66.9922%|6.8359%|92.9765%|784.933918|
|30|True|100.0000%|0.0000%|0.0000%|61.3766%|1080.749804|
|31|False|34.3750%|53.3203%|12.3047%|92.8933%|784.573292|
|32|False|41.2109%|43.1641%|15.6250%|92.9631%|792.473007|
|33|False|41.0156%|42.1875%|16.7969%|92.8310%|802.929725|
|34|False|35.9375%|42.3828%|21.6797%|92.8603%|790.643307|
|35|False|41.0156%|31.2500%|27.7344%|92.7940%|799.024181|

## B: periodic same-data probe comparisons

Train is the original unresampled distribution. Each pre/post pair uses the same trajectories and labels.

| Outer | Head / distribution | BA before→after | Ordinary accuracy before→after | Trajectory BCE before→after | Mixed BCE before→after |
|---|---|---:|---:|---:|---:|
|10|ema / train|65.0485%→69.8305%|71.1719%→73.6719%|0.619676→0.512161|0.700241→0.610750|
|10|ema / test|52.2891%→54.7341%|62.1094%→61.7188%|0.972716→0.830900|0.903006→0.792370|
|10|raw / train|52.0227%→77.9052%|63.9844%→78.0469%|0.907924→0.396518|0.951835→0.519267|
|10|raw / test|49.3835%→59.6068%|61.5234%→58.7891%|1.299927→0.658834|1.217350→0.798863|
|20|ema / train|66.9601%→68.1460%|65.1172%→66.1328%|0.415928→0.409803|0.703171→0.686222|
|20|ema / test|53.1843%→53.6887%|46.8750%→46.8750%|0.720116→0.716683|1.045489→1.053752|
|20|raw / train|48.4179%→72.8079%|47.4609%→70.2734%|0.564980→0.397319|0.978678→0.620858|
|20|raw / test|50.8627%→53.8824%|40.2344%→46.6797%|0.833552→0.714609|1.166199→1.077880|
|30|ema / train|48.0166%→56.6844%|47.3047%→53.7891%|0.588031→0.472792|1.018983→0.863656|
|30|ema / test|54.4652%→56.7986%|42.9688%→42.5781%|0.857849→0.732588|1.122567→1.166710|
|30|raw / train|37.4342%→70.3483%|43.2031%→64.9219%|1.090439→0.406445|1.202232→0.723369|
|30|raw / test|49.6650%→54.0027%|53.9062%→43.5547%|1.411874→0.703522|0.868772→1.102523|

Actual stored global probe loss rows: 470.
QA source records, full paired metrics, recall/confusion matrices, balancing statistics and clocks are retained in the JSON companion.

Required final scope remains both200/no201, all scheduled QA and final51713, saved/recoverable state and owned resource closure.
This partial report does not certify completion or recommend changing the experiment.
