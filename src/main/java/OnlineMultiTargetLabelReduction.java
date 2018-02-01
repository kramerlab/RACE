import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Attribute;
import weka.core.Instances;
import mulan.classifier.MultiLabelLearner;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.MacroSpecificity;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.SubsetAccuracy;
import mulan.transformations.RemoveAllLabels;


public class OnlineMultiTargetLabelReduction {
	
	static String[][] betaStr; 
	
	static ArrayList<Measure> measures = new ArrayList<Measure>();
	static ArrayList<MultiLabelInstances> batchesOfMultiLabelData; 
	
	static ArrayList<Integer> Tp = new ArrayList<Integer>(), Tn = new ArrayList<Integer>(), Fp = new ArrayList<Integer>(), Fn = new ArrayList<Integer>();
	
	public static MultiTargetLRUpdateable LR;

	static double[][] IW; 
	static double[] bias;
	public OnlineMultiTargetLabelReduction(double[][] IW , double[] bias){
		this.IW = IW;
		this.bias = bias;
	}
	public OnlineMultiTargetLabelReduction(){
		
	}
	
	public static void makeBetaString(double[][] curB){
		for(int i=0; i<curB.length; i++){
			for(int j=0; j<curB[i].length; j++){
				betaStr[i][j] = betaStr[i][j]+","+String.format( "%.2f", curB[i][j]);
			}
		}
	}
	
	public static long runOnDataset(boolean gram, MultiLabelInstances dataset, String xmlPath, String newXmlPath, String threshPath, int NumHiddenNeuron, 
									ArrayList<Measure> measures, int window, int N0, UpdateableClassifier classifier, 
									String activationFun, boolean adaptive, boolean shallowAE, double Hthresh, double Tthresh, double d) throws Exception{
				
		makeReducedDataset(dataset, xmlPath, newXmlPath, NumHiddenNeuron, N0, window);
		
		long t1 = System.currentTimeMillis();
		
		LR = new MultiTargetLRUpdateable(gram, classifier, measures.size(), dataset.getNumLabels(), NumHiddenNeuron, xmlPath, newXmlPath, activationFun, Hthresh, Tthresh,d);
		LR.build(batchesOfMultiLabelData.get(0));
		
		double[][] doubleBeta = LR.curBeta;
		betaStr = new String[doubleBeta.length][doubleBeta[0].length];
		makeBetaString(doubleBeta);
		
		///////////////
		int[] FNInLabels = new int[batchesOfMultiLabelData.get(0).getNumLabels()];
		int[] FPInLabels = new int[batchesOfMultiLabelData.get(0).getNumLabels()];
		int[] TPInLabels = new int[batchesOfMultiLabelData.get(0).getNumLabels()];
		int[] TNInLabels = new int[batchesOfMultiLabelData.get(0).getNumLabels()];
		PrintWriter pw = new PrintWriter(new File(threshPath));
		//////////////
		
		for(int i=1; i<batchesOfMultiLabelData.size(); i++){	
			System.out.println("#batch = "+i+" #size = "+batchesOfMultiLabelData.get(i).getNumInstances()+" #labels = "+batchesOfMultiLabelData.get(i).getNumLabels());
			Evaluator evaluator = new Evaluator();
			Evaluation eval = evaluator.evaluate((MultiLabelLearner) LR, batchesOfMultiLabelData.get(i), measures);
			
			///////////////
			Instances temp = batchesOfMultiLabelData.get(i).getDataSet();
			double[][] labels = LR.getWindowMatrix(LR.extractBatchLabels(batchesOfMultiLabelData.get(i)), false);
			double[][] preds = LR.makePredictionForBatch(temp);
			int countFP = 0, countFN = 0, countTP = 0, countTN = 0;
			
			for(int mk=0; mk<temp.numInstances();mk++){
				for(int mj=0; mj<labels[mk].length;mj++){
					if(labels[mk][mj] != preds[mk][mj]){
						if(labels[mk][mj] == 1){
							FNInLabels[mj]++;
							countFN++;
						}else{
							FPInLabels[mj]++;
							countFP++;
						}
					}else{
						if(labels[mk][mj] == 1){
							TPInLabels[mj]++;
							countTP++;
						}else{
							TNInLabels[mj]++;
							countTN++;
						}
					}
				}
			}
			pw.println("batch #"+i+" FN = "+countFN+" FP = "+countFP+" TP = "+countTP+" TN = "+countTN);
			//////////////
			
			LR.keepAllMeasures(eval.getMeasures());
			
			LR.updateClassifierForMLBatch(batchesOfMultiLabelData.get(i), adaptive, shallowAE);
			doubleBeta = LR.curBeta;
			makeBetaString(doubleBeta);
		}
		
		//////////////////
		pw.println("\n\nmisclassification of each label: ");
		for(int i=0; i<FNInLabels.length; i++)
			pw.println("label #"+i+" : FN = "+FNInLabels[i]+" FP = "+FPInLabels[i]+" TP = "+TPInLabels[i]+" TN = "+TNInLabels[i]);
		pw.close();
		/////////////////
		
		long t2 = System.currentTimeMillis();
		
		return (t2-t1);
	}
	
	
	public static long runOnTrainDataset(MultiLabelInstances train, MultiLabelInstances test, String xmlPath, String newXmlPath, 
			String threshPath, int NumHiddenNeuron,	ArrayList<Measure> measures, int window, int N0, UpdateableClassifier classifier, 
			String activationFun, boolean adaptive, boolean shallowAE, double Hthresh, double Tthresh, double d) throws Exception{

		makeReducedDataset(train, xmlPath, newXmlPath, NumHiddenNeuron, N0, window);
		
		long t1 = System.currentTimeMillis();
		
		LR = new MultiTargetLRUpdateable(IW, bias, classifier, measures.size(), train.getNumLabels(), NumHiddenNeuron, xmlPath, newXmlPath, activationFun, Hthresh, Tthresh,d);
		
		LR.build(batchesOfMultiLabelData.get(0));
		
		double[][] doubleBeta = LR.curBeta;
		betaStr = new String[doubleBeta.length][doubleBeta[0].length];
		makeBetaString(doubleBeta);
		
		PrintWriter pw = new PrintWriter(new File(threshPath));
		
		for(int i=1; i<batchesOfMultiLabelData.size(); i++){	
		System.out.println("#batch = "+i+" #size = "+batchesOfMultiLabelData.get(i).getNumInstances()+" #labels = "+batchesOfMultiLabelData.get(i).getNumLabels());
		Evaluator evaluator = new Evaluator();
		Evaluation eval = evaluator.evaluate((MultiLabelLearner) LR, test, measures);
		
		LR.keepAllMeasures(eval.getMeasures());
		
		LR.updateClassifierForMLBatch(batchesOfMultiLabelData.get(i), adaptive, shallowAE);
		doubleBeta = LR.curBeta;
		makeBetaString(doubleBeta);
		}
		
		long t2 = System.currentTimeMillis();
		
		return (t2-t1);
}
	
	
	public static long reconstructionError(boolean gram, MultiLabelInstances dataset, String xmlPath, String newXmlPath, 
			String threshPath, int NumHiddenNeuron,	ArrayList<Measure> measures, int window, int N0, UpdateableClassifier classifier, 
			String activationFun, boolean adaptive, boolean shallowAE, double Hthresh, double Tthresh, double d) throws Exception{

		makeReducedDataset(dataset, xmlPath, newXmlPath, NumHiddenNeuron, N0, window);
		
		long t1 = System.currentTimeMillis();
		
		LR = new MultiTargetLRUpdateable(gram, classifier, measures.size(), dataset.getNumLabels(), NumHiddenNeuron, xmlPath, newXmlPath, activationFun, Hthresh, Tthresh,d);
		
		LR.build(batchesOfMultiLabelData.get(0));
		
		double[][] doubleBeta = LR.curBeta;
		betaStr = new String[doubleBeta.length][doubleBeta[0].length];
		makeBetaString(doubleBeta);
		
		PrintWriter pw = new PrintWriter(new File(threshPath));
		
		for(int i=1; i<batchesOfMultiLabelData.size(); i++){	
		System.out.println("#batch = "+i+" #size = "+batchesOfMultiLabelData.get(i).getNumInstances()+" #labels = "+batchesOfMultiLabelData.get(i).getNumLabels());
		
		LR.updateClassifierForMLBatch(batchesOfMultiLabelData.get(i), adaptive, shallowAE);
		doubleBeta = LR.curBeta;
		makeBetaString(doubleBeta);
		
		Evaluator evaluator = new Evaluator();
		Evaluation eval = evaluator.evaluate((MultiLabelLearner) LR, batchesOfMultiLabelData.get(i), measures);
		LR.keepAllMeasures(eval.getMeasures());
		
		}
		
		long t2 = System.currentTimeMillis();
		
		return (t2-t1);
}
	
	
	public static void makeReducedDataset(MultiLabelInstances dataset, String xmlPath, String newXmlPath, int hiddenNeurons, int N0, int window) throws Exception{
		
		//new xml description 
		File f = new File(newXmlPath);
		if(!f.exists())
			makeXmlDescription(newXmlPath, hiddenNeurons);
		
		//make dataset windowed
		batchesOfMultiLabelData = MakeDataWindowed(dataset, xmlPath, N0, window);

	}


	public static void makeXmlDescription(String xmlPath, int hiddenNeurons) throws FileNotFoundException{
		PrintWriter pw = new PrintWriter(new File(xmlPath));
		pw.println("<?xml version=\"1.0\" encoding=\"utf-8\"?> \n <labels xmlns=\"http://mulan.sourceforge.net/labels\">");
		
		for(int i=0; i<hiddenNeurons; i++){
			pw.println("<label name=\"hiddenLabel_"+i+"\"></label>");
		}
		
		pw.print("</labels>");
		pw.close();
	}


	//make one MultiLabelInstances dataset windowed by size length
	public static ArrayList<MultiLabelInstances> MakeDataWindowed(MultiLabelInstances multiInst, String xmlPath, int N0, int length) throws InvalidDataFormatException{
		ArrayList<MultiLabelInstances> array = new ArrayList<>();
		Instances inst = multiInst.getDataSet();
		
		array.add(new MultiLabelInstances(getWindow(inst, 0, N0), xmlPath));
		
		for(int ins=N0; ins<inst.numInstances();ins = ins+length){
			if(ins+length<inst.numInstances()){
				array.add(new MultiLabelInstances(getWindow(inst, ins, ins+length), xmlPath));
			}else{
				array.add(new MultiLabelInstances(getWindow(inst, ins, inst.numInstances()), xmlPath));
			}
		}
		
		return array;
	}
	
	//make one Instances dataset windowed by size length
		public static ArrayList<Instances> MakeDataWindowed(Instances inst, int N0, int length){
			ArrayList<Instances> array = new ArrayList<>();
			
			array.add(getWindow(inst, 0, N0));
			
			for(int ins=N0; ins<inst.numInstances();ins = ins+length){
				if(ins+length<inst.numInstances()){
					array.add(getWindow(inst, ins, ins+length));
				}else{
					array.add(getWindow(inst, ins, inst.numInstances()));
				}
			}
			
			return array;
		}
		
		
		public static Instances getWindow(Instances inst, int start, int end){
			Instances window = new Instances(inst);
			window.delete();
			
			for(int i=start; i < end; i++)
				window.add(inst.instance(i));
			
			return window;
		}

}
