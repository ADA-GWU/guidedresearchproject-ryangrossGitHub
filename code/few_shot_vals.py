input = [[0.2578300000000013, 0.00015740740740736957, 0.26, 0.33499999999999996],
[0.25334988003664305, 0.01259551859089609, 0.26, 0.33497222222222217],
[0.24882519999999886, 0.025034444444443964, 0.26333333333333336, 0.33611111111111114],
[0.2441824727765322, 0.037393670135153025, 0.26, 0.3388888888888889],
[0.23969879999999932, 0.049769259259260824, 0.26333333333333336, 0.3336111111111111],
[0.23511127278051447, 0.06237380481721751, 0.26333333333333336, 0.33444444444444443],
[0.23053181823340196, 0.07495986542552555, 0.26666666666666666, 0.3383333333333333],
[0.22746690006224526, 0.08879737660708065, 0.26999999999999996, 0.2980787037037037],
[0.22502308653885167, 0.1033888897083544, 0.2733333333333333, 0.29215608465608467],
[0.2222352000000029, 0.11784074074074022, 0.27666666666666667, 0.30027777777777775],
[0.21971186674083185, 0.13184722221089154, 0.25666666666666665, 0.24754629629629632],
[0.21995860006547047, 0.14658555555266653, 0.26999999999999996, 0.2475],
[0.22000189097046813, 0.1615220538736619, 0.26999999999999996, 0.2505808080808081],
[0.2202247999999969, 0.17661185185185158, 0.2733333333333333, 0.245],
[0.22043156370676228, 0.19178744105792173, 0.26666666666666666, 0.2366919191919192],
[0.22080106792590187, 0.2057591358582222, 0.25, 0.25083333333333335],
[0.22107934552494726, 0.21985478112353685, 0.26333333333333336, 0.2361111111111111],
[0.22281879999999604, 0.2338177777777761, 0.25666666666666665, 0.20083333333333334]]
pred = [[0.2245582544750442, 0.24778077443201535, 0.25, 0.20083333333333334],
[0.22629770895009236, 0.2617437710862546, 0.24333333333333332, 0.20083333333333334],
[0.22803716342514052, 0.2757067677404939, 0.23666666666666666, 0.20083333333333334],
[0.22977661790018868, 0.2896697643947332, 0.23, 0.20083333333333334],
[0.23151607237523684, 0.3036327610489725, 0.22333333333333333, 0.20083333333333334],
[0.233255526850285, 0.3175957577032118, 0.21666666666666667, 0.20083333333333334],
[0.23499498132533316, 0.3315587543574511, 0.21, 0.20083333333333334],
[0.23673443580038132, 0.3455217510116904, 0.20333333333333334, 0.20083333333333334],
[0.23847389027542948, 0.3594847476659297, 0.19666666666666666, 0.20083333333333334],
[0.24021334475047764, 0.373447744320169, 0.19, 0.20083333333333334],
[0.2419527992255258, 0.3874107409744083, 0.18333333333333332, 0.20083333333333334],
[0.24369225370057396, 0.4013737376286476, 0.17666666666666667, 0.20083333333333334],
[0.24543170817562212, 0.4153367342828869, 0.17, 0.20083333333333334],
[0.24717116265067028, 0.4292997309371262, 0.16333333333333333, 0.20083333333333334],
[0.24891061712571844, 0.4432627275913655, 0.15666666666666668, 0.20083333333333334],
[0.2506500716007666, 0.4572257242456048, 0.15, 0.20083333333333334],
[0.25238952607581476, 0.471188720, 0, 0],
[0.25238952607581476, 0.471188720, 0, 0]]
actuals = [[0.22790716368971004, 0.26016457906457646, 0.24333333333333332, 0.1963888888888889],
[0.23061360006119286, 0.2729801542530013, 0.24666666666666667, 0.1966435185185185],
[0.2337508000000014, 0.2864488888888865, 0.2733333333333333, 0.18555555555555556],
[0.2372240000000005, 0.300328518518517, 0.26999999999999996, 0.19055555555555553],
[0.24023999999999718, 0.31427222222222234, 0.26333333333333336, 0.19833333333333336],
[0.24255766672850995, 0.32749033946157513, 0.24027777777777778, 0.2147222222222222],
[0.2439099999999968, 0.3407481481481479, 0.24333333333333332, 0.22555555555555556],
[0.24496479999999962, 0.35432037037037084, 0.25, 0.23111111111111113],
[0.24573800000000007, 0.3683303703703702, 0.26, 0.2388888888888889],
[0.2464444800502122, 0.38288455553627343, 0.26, 0.22836111111111113],
[0.2476211600591313, 0.3971245555274662, 0.25666666666666665, 0.22586111111111112],
[0.24886700004935564, 0.41135696294031293, 0.26, 0.2263888888888889],
[0.2500867999999997, 0.425774814814815, 0.26, 0.22694444444444445],
[0.2512216364191204, 0.4399827609207682, 0.2533333333333333, 0.22724747474747475],
[0.2525945200498626, 0.4540227777489347, 0.26333333333333336, 0.2207777777777778],
[0.2541882334134698, 0.4687455246458434, 0.2733333333333333, 0.22245370370370368],
[0.2556348000000014, 0.4835870370370373, 0.26999999999999996, 0.22555555555555556],
[0.2567914546177178, 0.49841710434907066, 0.26999999999999996, 0.23141414141414143]]