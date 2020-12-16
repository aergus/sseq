var Module=typeof pyodide._module!=="undefined"?pyodide._module:{};Module.checkABI(1);if(!Module.expectedDataFileDownloads){Module.expectedDataFileDownloads=0;Module.finishedDataFileDownloads=0}Module.expectedDataFileDownloads++;(function(){var loadPackage=function(metadata){var PACKAGE_PATH;if(typeof window==="object"){PACKAGE_PATH=window["encodeURIComponent"](window.location.pathname.toString().substring(0,window.location.pathname.toString().lastIndexOf("/"))+"/")}else if(typeof location!=="undefined"){PACKAGE_PATH=encodeURIComponent(location.pathname.toString().substring(0,location.pathname.toString().lastIndexOf("/"))+"/")}else{throw"using preloaded data can only be done on a web page or in a web worker"}var PACKAGE_NAME="html5lib.data";var REMOTE_PACKAGE_BASE="html5lib.data";if(typeof Module["locateFilePackage"]==="function"&&!Module["locateFile"]){Module["locateFile"]=Module["locateFilePackage"];err("warning: you defined Module.locateFilePackage, that has been renamed to Module.locateFile (using your locateFilePackage for now)")}var REMOTE_PACKAGE_NAME=Module["locateFile"]?Module["locateFile"](REMOTE_PACKAGE_BASE,""):REMOTE_PACKAGE_BASE;var REMOTE_PACKAGE_SIZE=metadata.remote_package_size;var PACKAGE_UUID=metadata.package_uuid;function fetchRemotePackage(packageName,packageSize,callback,errback){var xhr=new XMLHttpRequest;xhr.open("GET",packageName,true);xhr.responseType="arraybuffer";xhr.onprogress=function(event){var url=packageName;var size=packageSize;if(event.total)size=event.total;if(event.loaded){if(!xhr.addedTotal){xhr.addedTotal=true;if(!Module.dataFileDownloads)Module.dataFileDownloads={};Module.dataFileDownloads[url]={loaded:event.loaded,total:size}}else{Module.dataFileDownloads[url].loaded=event.loaded}var total=0;var loaded=0;var num=0;for(var download in Module.dataFileDownloads){var data=Module.dataFileDownloads[download];total+=data.total;loaded+=data.loaded;num++}total=Math.ceil(total*Module.expectedDataFileDownloads/num);if(Module["setStatus"])Module["setStatus"]("Downloading data... ("+loaded+"/"+total+")")}else if(!Module.dataFileDownloads){if(Module["setStatus"])Module["setStatus"]("Downloading data...")}};xhr.onerror=function(event){throw new Error("NetworkError for: "+packageName)};xhr.onload=function(event){if(xhr.status==200||xhr.status==304||xhr.status==206||xhr.status==0&&xhr.response){var packageData=xhr.response;callback(packageData)}else{throw new Error(xhr.statusText+" : "+xhr.responseURL)}};xhr.send(null)}function handleError(error){console.error("package error:",error)}var fetchedCallback=null;var fetched=Module["getPreloadedPackage"]?Module["getPreloadedPackage"](REMOTE_PACKAGE_NAME,REMOTE_PACKAGE_SIZE):null;if(!fetched)fetchRemotePackage(REMOTE_PACKAGE_NAME,REMOTE_PACKAGE_SIZE,function(data){if(fetchedCallback){fetchedCallback(data);fetchedCallback=null}else{fetched=data}},handleError);function runWithFS(){function assert(check,msg){if(!check)throw msg+(new Error).stack}Module["FS_createPath"]("/","lib",true,true);Module["FS_createPath"]("/lib","python3.8",true,true);Module["FS_createPath"]("/lib/python3.8","site-packages",true,true);Module["FS_createPath"]("/lib/python3.8/site-packages","html5lib",true,true);Module["FS_createPath"]("/lib/python3.8/site-packages/html5lib","filters",true,true);Module["FS_createPath"]("/lib/python3.8/site-packages/html5lib","treeadapters",true,true);Module["FS_createPath"]("/lib/python3.8/site-packages/html5lib","treebuilders",true,true);Module["FS_createPath"]("/lib/python3.8/site-packages/html5lib","treewalkers",true,true);Module["FS_createPath"]("/lib/python3.8/site-packages/html5lib","_trie",true,true);Module["FS_createPath"]("/lib/python3.8/site-packages","html5lib-1.0.1-py3.8.egg-info",true,true);function DataRequest(start,end,audio){this.start=start;this.end=end;this.audio=audio}DataRequest.prototype={requests:{},open:function(mode,name){this.name=name;this.requests[name]=this;Module["addRunDependency"]("fp "+this.name)},send:function(){},onload:function(){var byteArray=this.byteArray.subarray(this.start,this.end);this.finish(byteArray)},finish:function(byteArray){var that=this;Module["FS_createPreloadedFile"](this.name,null,byteArray,true,true,function(){Module["removeRunDependency"]("fp "+that.name)},function(){if(that.audio){Module["removeRunDependency"]("fp "+that.name)}else{err("Preloading file "+that.name+" failed")}},false,true);this.requests[this.name]=null}};function processPackageData(arrayBuffer){Module.finishedDataFileDownloads++;assert(arrayBuffer,"Loading data file failed.");assert(arrayBuffer instanceof ArrayBuffer,"bad input to processPackageData");var byteArray=new Uint8Array(arrayBuffer);var curr;var compressedData={data:null,cachedOffset:234557,cachedIndexes:[-1,-1],cachedChunks:[null,null],offsets:[0,1029,1771,2548,3285,3960,4610,5471,6071,6800,7947,8984,10114,11275,12165,13110,14006,14910,15820,16721,17659,18572,19514,20370,21297,22234,23144,24005,24941,25846,26753,27687,28469,29323,30265,31195,32138,32990,33901,34821,35666,36905,37917,39155,40252,41113,42164,43523,44591,45690,46624,47050,47718,48461,49262,50190,51187,52013,52920,53904,54813,55843,56887,57657,58503,59174,60020,60842,61658,62578,63509,64203,65225,66194,67023,67868,68860,69909,70633,71581,72490,73471,74365,75166,76126,76852,77802,78780,79658,80623,81284,82120,83076,83967,84882,85765,86594,87368,87995,89148,90369,91615,92785,93946,94940,95837,96592,97834,99088,100420,101687,103140,104688,106133,107094,108197,109391,110427,111794,112941,114103,115219,116603,117702,118595,119651,120659,121787,122717,123829,124775,125891,127191,128018,129213,130158,131234,132027,132531,133286,134109,134715,135342,135998,136603,137083,137733,138436,138893,139577,140415,141344,142043,142642,143391,144306,145195,145667,146259,146947,147522,148335,149042,149583,150153,150718,151320,151842,152790,154221,155433,156740,157776,158840,159554,160567,161469,162304,163147,163917,164763,165215,165723,166450,167187,167904,168668,169534,170587,171433,172548,173648,174793,176013,177147,178285,179432,180531,181622,182708,183794,184723,185898,186835,187801,188765,189596,190723,191642,192523,193345,194220,195161,196453,197410,198281,199341,200098,201082,202266,203557,204864,206180,207140,208118,209013,210257,211242,212166,213072,214147,215255,216269,217550,218602,219542,220476,221656,222616,223877,225182,226500,227838,229089,230383,231649,232663,233563,234090,234423],sizes:[1029,742,777,737,675,650,861,600,729,1147,1037,1130,1161,890,945,896,904,910,901,938,913,942,856,927,937,910,861,936,905,907,934,782,854,942,930,943,852,911,920,845,1239,1012,1238,1097,861,1051,1359,1068,1099,934,426,668,743,801,928,997,826,907,984,909,1030,1044,770,846,671,846,822,816,920,931,694,1022,969,829,845,992,1049,724,948,909,981,894,801,960,726,950,978,878,965,661,836,956,891,915,883,829,774,627,1153,1221,1246,1170,1161,994,897,755,1242,1254,1332,1267,1453,1548,1445,961,1103,1194,1036,1367,1147,1162,1116,1384,1099,893,1056,1008,1128,930,1112,946,1116,1300,827,1195,945,1076,793,504,755,823,606,627,656,605,480,650,703,457,684,838,929,699,599,749,915,889,472,592,688,575,813,707,541,570,565,602,522,948,1431,1212,1307,1036,1064,714,1013,902,835,843,770,846,452,508,727,737,717,764,866,1053,846,1115,1100,1145,1220,1134,1138,1147,1099,1091,1086,1086,929,1175,937,966,964,831,1127,919,881,822,875,941,1292,957,871,1060,757,984,1184,1291,1307,1316,960,978,895,1244,985,924,906,1075,1108,1014,1281,1052,940,934,1180,960,1261,1305,1318,1338,1251,1294,1266,1014,900,527,333,134],successes:[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]};compressedData.data=byteArray;assert(typeof Module.LZ4==="object","LZ4 not present - was your app build with  -s LZ4=1  ?");Module.LZ4.loadPackage({metadata:metadata,compressedData:compressedData});Module["removeRunDependency"]("datafile_html5lib.data")}Module["addRunDependency"]("datafile_html5lib.data");if(!Module.preloadResults)Module.preloadResults={};Module.preloadResults[PACKAGE_NAME]={fromCache:false};if(fetched){processPackageData(fetched);fetched=null}else{fetchedCallback=processPackageData}}if(Module["calledRun"]){runWithFS()}else{if(!Module["preRun"])Module["preRun"]=[];Module["preRun"].push(runWithFS)}};loadPackage({files:[{start:0,audio:0,end:83518,filename:"/lib/python3.8/site-packages/html5lib/constants.py"},{start:83518,audio:0,end:202469,filename:"/lib/python3.8/site-packages/html5lib/html5parser.py"},{start:202469,audio:0,end:218215,filename:"/lib/python3.8/site-packages/html5lib/serializer.py"},{start:218215,audio:0,end:234920,filename:"/lib/python3.8/site-packages/html5lib/_ihatexml.py"},{start:234920,audio:0,end:267419,filename:"/lib/python3.8/site-packages/html5lib/_inputstream.py"},{start:267419,audio:0,end:343987,filename:"/lib/python3.8/site-packages/html5lib/_tokenizer.py"},{start:343987,audio:0,end:347990,filename:"/lib/python3.8/site-packages/html5lib/_utils.py"},{start:347990,audio:0,end:349135,filename:"/lib/python3.8/site-packages/html5lib/__init__.py"},{start:349135,audio:0,end:350054,filename:"/lib/python3.8/site-packages/html5lib/filters/alphabeticalattributes.py"},{start:350054,audio:0,end:350340,filename:"/lib/python3.8/site-packages/html5lib/filters/base.py"},{start:350340,audio:0,end:353285,filename:"/lib/python3.8/site-packages/html5lib/filters/inject_meta_charset.py"},{start:353285,audio:0,end:356916,filename:"/lib/python3.8/site-packages/html5lib/filters/lint.py"},{start:356916,audio:0,end:367504,filename:"/lib/python3.8/site-packages/html5lib/filters/optionaltags.py"},{start:367504,audio:0,end:393740,filename:"/lib/python3.8/site-packages/html5lib/filters/sanitizer.py"},{start:393740,audio:0,end:394954,filename:"/lib/python3.8/site-packages/html5lib/filters/whitespace.py"},{start:394954,audio:0,end:394954,filename:"/lib/python3.8/site-packages/html5lib/filters/__init__.py"},{start:394954,audio:0,end:396669,filename:"/lib/python3.8/site-packages/html5lib/treeadapters/genshi.py"},{start:396669,audio:0,end:398445,filename:"/lib/python3.8/site-packages/html5lib/treeadapters/sax.py"},{start:398445,audio:0,end:399095,filename:"/lib/python3.8/site-packages/html5lib/treeadapters/__init__.py"},{start:399095,audio:0,end:413662,filename:"/lib/python3.8/site-packages/html5lib/treebuilders/base.py"},{start:413662,audio:0,end:422497,filename:"/lib/python3.8/site-packages/html5lib/treebuilders/dom.py"},{start:422497,audio:0,end:435249,filename:"/lib/python3.8/site-packages/html5lib/treebuilders/etree.py"},{start:435249,audio:0,end:449371,filename:"/lib/python3.8/site-packages/html5lib/treebuilders/etree_lxml.py"},{start:449371,audio:0,end:452963,filename:"/lib/python3.8/site-packages/html5lib/treebuilders/__init__.py"},{start:452963,audio:0,end:460439,filename:"/lib/python3.8/site-packages/html5lib/treewalkers/base.py"},{start:460439,audio:0,end:461852,filename:"/lib/python3.8/site-packages/html5lib/treewalkers/dom.py"},{start:461852,audio:0,end:466390,filename:"/lib/python3.8/site-packages/html5lib/treewalkers/etree.py"},{start:466390,audio:0,end:472687,filename:"/lib/python3.8/site-packages/html5lib/treewalkers/etree_lxml.py"},{start:472687,audio:0,end:474996,filename:"/lib/python3.8/site-packages/html5lib/treewalkers/genshi.py"},{start:474996,audio:0,end:480710,filename:"/lib/python3.8/site-packages/html5lib/treewalkers/__init__.py"},{start:480710,audio:0,end:481876,filename:"/lib/python3.8/site-packages/html5lib/_trie/datrie.py"},{start:481876,audio:0,end:483639,filename:"/lib/python3.8/site-packages/html5lib/_trie/py.py"},{start:483639,audio:0,end:484569,filename:"/lib/python3.8/site-packages/html5lib/_trie/_base.py"},{start:484569,audio:0,end:484858,filename:"/lib/python3.8/site-packages/html5lib/_trie/__init__.py"},{start:484858,audio:0,end:484859,filename:"/lib/python3.8/site-packages/html5lib-1.0.1-py3.8.egg-info/dependency_links.txt"},{start:484859,audio:0,end:503628,filename:"/lib/python3.8/site-packages/html5lib-1.0.1-py3.8.egg-info/PKG-INFO"},{start:503628,audio:0,end:503899,filename:"/lib/python3.8/site-packages/html5lib-1.0.1-py3.8.egg-info/requires.txt"},{start:503899,audio:0,end:510310,filename:"/lib/python3.8/site-packages/html5lib-1.0.1-py3.8.egg-info/SOURCES.txt"},{start:510310,audio:0,end:510319,filename:"/lib/python3.8/site-packages/html5lib-1.0.1-py3.8.egg-info/top_level.txt"}],remote_package_size:238653,package_uuid:"0e191a6b-42b7-492a-92ee-18e41920f8fc"})})();