waitForUser("Select the first image to analyze");
open();
title = getTitle(); 
selectWindow(title);
//run("Duplicate...", "title=duplicate");
run("Trainable Weka Segmentation");
//selectWindow("Trainable Weka Segmentation v3.2.33");
//get path
waitForUser("Hit Load Classifier, select the classifer, and do not close this popup until after");
//call("trainableSegmentation.Weka_Segmentation.loadClassifier", "/Users/students/Downloads/spetinWEKAclassifier.model");
waitForUser("Click ok if classifier is loading");
loadclass = getBoolean("Has the classifier fully loaded?");
while (loadclass != 1){
	loadclass = getBoolean("Has the classifier fully loaded?");
}
waitForUser("Ready?"); 
call("trainableSegmentation.Weka_Segmentation.getResult");
selectWindow("Classified image");
setAutoThreshold("Intermodes");
//run("Threshold...");
//setThreshold(1, 255);
run("Convert to Mask");
roiManager("Deselect");
run("Analyze Particles...", "size=0.10-Infinity show=Outlines display exclude clear add");
waitForUser("Choose a save folder");
path = getDirectory("Choose a Save Directory"); 
var n = nResults;
for (i=0;i<n;i++){
	//Name of super res image
	selectWindow(title); 
	roiManager("Select", i);
	run("Duplicate...", "title="+title+i);
	var width = getWidth();
	var height = getHeight();
	if (width + height < 200){
		run("Canvas Size...", "width=100 height=100 position=Center zero");
	}
	//Define path
	saveAs("Tiff", path+title+i+".tif");
	close();
}
close("Classified image");
close("Drawing of Classification result");
con = getBoolean("Analyze another image?");
while (con == 1){
	waitForUser("Apply Classifier", "Hit Apply Classifier, select an image, select NO for probability map, and do not close the popup until after");
	//call("trainableSegmentation.Weka_Segmentation.applyClassifier", newimg, title, "showResults=true", "storeResults=false", "probabilityMaps=false", "")
	waitForUser("Click ok if the classifier is being applied.");
	appclass = getBoolean("Has there been a classification result yet?");
	while (appclass != 1){
		appclass = getBoolean("Has there been a classification result yet?");
	}
	title = getTitle();
	selectWindow("Classification result");
	setAutoThreshold("Intermodes");
	//run("Threshold...");
	//setThreshold(1, 255);
	run("Convert to Mask");
	roiManager("Deselect");
	run("Analyze Particles...", "size=0.10-Infinity show=Outlines display exclude clear add");
	var n = nResults;
	waitForUser("Choose a save folder");
	path = getDirectory("Choose a Save Directory"); 
	for (i=0;i<n;i++){
		//Name of super res image
		selectWindow(title); 
		roiManager("Select", i);
		run("Duplicate...", "title="+title+i);
		var width = getWidth();
		var height = getHeight();
		if (width + height < 200){
			run("Canvas Size...", "width=100 height=100 position=Center zero");
		}
		saveAs("Tiff", path+title+i+".tif");
		close();
	}
	close("Classification result");
	con = getBoolean("Analyze another image?");
	}
end = getBoolean("Exit the application?");
while (end !=1){
	getBoolean("Macro language cant do anything from here, you must hit yes");
}
run("Close All");


	 
