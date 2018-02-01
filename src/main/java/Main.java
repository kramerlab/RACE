import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;

import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.MacroSpecificity;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.MicroSpecificity;
import mulan.evaluation.measure.SubsetAccuracy;

import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.functions.SGD;
import weka.core.Instances;

/**
 * The main class for doing the experiments
 *
 */

public class Main {
	static Random rand = new Random();
		
	public static ArrayList<Instances> MakeDataWindowed(Instances inst, int N0, int length){
		ArrayList<Instances> array = new ArrayList<Instances>();
		
		Instances sub = new Instances(inst);
		for(int ins=inst.numInstances()-1; ins>=N0; ins--){
			sub.delete(ins);
		}
		array.add(sub);
		for(int ins=N0; ins<inst.numInstances();ins = ins+length){
			Instances subi = new Instances(inst);
			subi.delete();
			if(ins+length<inst.numInstances()){
				for(int i=ins; i<ins+length; i++){
					subi.add(inst.instance(i));
				}
			}else{
				for(int i=ins; i<inst.numInstances(); i++){
					subi.add(inst.instance(i));
				}
			}
			array.add(subi);
		}
		
		return array;
	}


	
	public static void main(String[] args) throws Exception {
		String datasetPath = "data/CAL500/CAL500", datasetName = "CAL500" , algorithm = "RACE",
				outPath = "results/", run = "1"; //outpath includes / at the end
		int NumHidden = -1;
		int windowSize = -1;
		String hiddenL = "log";
		boolean first = false;
		//========= the parameters I dont give the option to be set in input ==========
		UpdateableClassifier uclassifier = new NaiveBayesUpdateable();
		//new NaiveBayesMultinomialUpdateable(), new HoeffdingTree(), new SGD()
		String activationFunc = "HardLim";
		double hardLimThresh = 0;
		double testThresh = 0/*-1, -0.5, 0, 0.5, 1*/;
		double delta = 0.1; 
		int iter = 5;
		//==============================================================================
		
		for(int i=0; i<args.length; i++){
			if(args[i].equals("--dataset")){
				datasetPath = args[++i];
				StringTokenizer stg = new StringTokenizer(datasetPath, "/");
				while(stg.hasMoreTokens()){
					datasetName = stg.nextToken();
				}
			}
			if(args[i].equals("--method"))
				algorithm = args[++i];
			if(args[i].equals("--compress"))
				hiddenL = args[++i];
			if(args[i].equals("--hiddenNeuron"))
				NumHidden = new Integer(args[++i]);
			if(args[i].equals("--windowSize"))
				windowSize = new Integer(args[++i]);
			if(args[i].equals("--outputPath"))
				outPath = args[++i];
			if(args[i].equals("--run"))
				run = args[++i];
			if(args[i].equals("--label"))
				first = new Boolean(args[++i]);
			if(args[i].equals("--iter"))
				iter = new Integer(args[++i]);
		}
		
		MultiLabelInstances dataset = new MultiLabelInstances(datasetPath+".arff", datasetPath+".xml");
		if(datasetPath.contains("nus-wide")){
			Instances inst = new Instances(new FileReader(datasetPath+".arff"));
			inst.deleteAttributeAt(0);
			dataset = new MultiLabelInstances(inst, datasetPath+".xml");
		}
		
		if(NumHidden == -1){		//if NumHidden is not set as the input parameter
			//previous method: the multiplicatives of 5 with minimum of 10
			if(hiddenL.equals("linear")){
				double q = (double)dataset.getNumLabels()/10;
				int q5 = (int) Math.ceil(q/5);
				if(q5*5 > 10)
					NumHidden = q5*5;
				else 
					NumHidden = 10;				
			}else if(hiddenL.equals("log"))	
				NumHidden = (int)Math.ceil(Math.log((double)dataset.getNumLabels())/Math.log(2));
		}
		
		if(windowSize == -1){		//if windowSize is not set as the input parameter
			if(datasetPath.contains("CAL500"))
				windowSize = 50;
			else if(dataset.getNumInstances() <= 5000)
				windowSize = 100;
			else
				windowSize = 500;			
		}						
		
		ArrayList<Measure> measures = new ArrayList<Measure>();	
		measures.add(new AveragePrecision());
		measures.add(new Coverage());
		measures.add(new SubsetAccuracy());	
		measures.add(new HammingLoss());
		
		measures.add(new ExampleBasedAccuracy());
		measures.add(new ExampleBasedSpecificity());
		measures.add(new ExampleBasedPrecision());
		measures.add(new ExampleBasedRecall());
		measures.add(new ExampleBasedFMeasure());
		
		measures.add(new MicroAUC(dataset.getNumLabels()));
		measures.add(new MicroSpecificity(dataset.getNumLabels()));
		measures.add(new MicroPrecision(dataset.getNumLabels()));
		measures.add(new MicroRecall(dataset.getNumLabels()));
		measures.add(new MicroFMeasure(dataset.getNumLabels()));
		
		measures.add(new myMacroAUC(dataset.getNumLabels()));
		measures.add(new MacroSpecificity(dataset.getNumLabels()));
		measures.add(new MacroPrecision(dataset.getNumLabels()));
		measures.add(new MacroRecall(dataset.getNumLabels()));
		measures.add(new MacroFMeasure(dataset.getNumLabels()));
			
//===============================================================================================	
		if(dataset.getNumInstances()/windowSize > 3){  //at least 4 batches of data
			
			String classifierName = uclassifier.getClass().getName();
			classifierName = classifierName.substring(classifierName.lastIndexOf(".")+1);
			String outputPath = outPath+datasetName+"_"+windowSize+"_"+classifierName+"/"+NumHidden+"/"+algorithm+"/";
			File f = new File(outputPath);
			f.mkdirs();
			
			int N0 = windowSize;
			ArrayList<Instances> dataBatch = MakeDataWindowed(dataset.getDataSet(), N0, windowSize);
			
			ArrayList<ArrayList<Double>[]> allmeasures = new ArrayList<ArrayList<Double>[]>();
			ArrayList<Long> times = new ArrayList<Long>(); 
		
			long t1,t2;
			
			switch(algorithm){
			case "RACE-regression": //regression - fix encoder - GS
				uclassifier = new SGD();
				((SGD)uclassifier).setOptions(new String[]{"-F", "2", "-N", "-L", "0.0001"});
				String[] opt = ((SGD)uclassifier).getOptions();
				for(int i=0; i<opt.length; i++){
					System.out.println(opt[i]);
				}
				activationFunc = "No";
				if(dataset.getNumLabels() > NumHidden){
					System.out.println("RACE - "+NumHidden+ " - "+false);
					String newXmlPath = outPath+"reducedXML/Reduced_"+datasetName+"_"+NumHidden+".xml";
					f = new File(outPath+"reducedXML/");
					if(!f.exists())
						f.mkdirs();
					OnlineMultiTargetLabelReduction OLR = new OnlineMultiTargetLabelReduction();
					long olrTime = OLR.runOnDataset(true, dataset, datasetPath+".xml", newXmlPath, outputPath+"misclass.txt", NumHidden, measures, windowSize, N0, uclassifier, activationFunc, false, false, hardLimThresh,testThresh,delta);
					allmeasures.add(OLR.LR.outputMeasures);
					times.add(olrTime);						
				}
				break;
			case "RACE-regression-autoencoder":  //regression - AE encoder - GS
				uclassifier = new SGD();
				((SGD)uclassifier).setOptions(new String[]{"-F", "2", "-N", "-L", "0.0001"});
				opt = ((SGD)uclassifier).getOptions();
				for(int i=0; i<opt.length; i++){
					System.out.println(opt[i]);
				}
				activationFunc = "No";
				if(dataset.getNumLabels() > NumHidden){
					System.out.println("RACE - "+NumHidden+ " - "+false);
					String newXmlPath = outPath+"reducedXML/Reduced_"+datasetName+"_"+NumHidden+".xml";
					f = new File(outPath+"reducedXML/");
					if(!f.exists())
						f.mkdirs();
					OnlineMultiTargetLabelReduction OLR = new OnlineMultiTargetLabelReduction();
					long olrTime = OLR.runOnDataset(true, dataset, datasetPath+".xml", newXmlPath, outputPath+"misclass.txt", NumHidden, measures, windowSize, N0, uclassifier, activationFunc, false, true, hardLimThresh,testThresh,delta);
					allmeasures.add(OLR.LR.outputMeasures);
					times.add(olrTime);						
				}
				break;
			case "RACE": //classification - fix encoder - GS
				if(dataset.getNumLabels() > NumHidden){
					System.out.println("RACE - "+NumHidden+ " - "+false);
					String newXmlPath = outPath+"reducedXML/Reduced_"+datasetName+"_"+NumHidden+".xml";
					f = new File(outPath+"reducedXML/");
					if(!f.exists())
						f.mkdirs();
					OnlineLabelReduction OLR = new OnlineLabelReduction();
					long olrTime = OLR.runOnDataset(first, true, dataset, datasetPath+".xml", newXmlPath, outputPath+"misclass.txt", NumHidden, measures, windowSize, N0, uclassifier, activationFunc, false, false, hardLimThresh,testThresh,delta);
					allmeasures.add(OLR.LR.outputMeasures);
					times.add(olrTime);						
				}
				break;
			case "RACE-autoencoder": //classification - AE encoder - GS
				if(dataset.getNumLabels() > NumHidden){
					System.out.println("RACE - "+NumHidden+ " - "+false);
					String newXmlPath = outPath+"reducedXML/Reduced_"+datasetName+"_"+NumHidden+".xml";
					f = new File(outPath+"reducedXML/");
					if(!f.exists())
						f.mkdirs();
					OnlineLabelReduction OLR = new OnlineLabelReduction();
					long olrTime = OLR.runOnDataset(first, true, dataset, datasetPath+".xml", newXmlPath, outputPath+"misclass.txt", NumHidden, measures, windowSize, N0, uclassifier, activationFunc, false, true, hardLimThresh,testThresh,delta);
					allmeasures.add(OLR.LR.outputMeasures);
					times.add(olrTime);						
				}
				break;
			default: break; 				
			}
			
//===============================================================================================	
			//write in output file
			String outputPath1 = outputPath+"time/";
			f = new File(outputPath1);
			f.mkdirs();
			PrintWriter pw = new PrintWriter(new File(outputPath1+run+".txt"));
			for(int i=0; i<times.size();i++)
				pw.print((double)times.get(i)/1000+" ");
			pw.close();
			
			int measureLength = (dataset.getNumInstances()-N0)/windowSize;
			
			for(int i=0; i<measures.size(); i++){ //over all measures
				outputPath1 = outputPath+measures.get(i).getName()+"/";
				f = new File(outputPath1);
				f.mkdirs();
				pw = new PrintWriter(new File(outputPath1+run+".txt"));

				for(int m=0; m<measureLength;m++){ //over number of windows
					for(int l=0; l<allmeasures.size();l++){ //over all algorithms
						pw.print((allmeasures.get(l))[i].get(m)+" ");
					}
					pw.println();
				}
				pw.close();		
			}

			//make the averages of all measures (the folder is outputPath)
			System.out.println(outputPath);
			if(outputPath.endsWith("/"))
				outputPath = outputPath.substring(0, outputPath.length()-1);
			new AverageOfMeasures().calculateAvgOfOneMethod(outputPath,false);
			
		}else{
			System.out.println("less than 3 batches of data!");
		}
		
	}

}
