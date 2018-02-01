import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import weka.classifiers.UpdateableClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;


public class MultiTargetLRUpdateable extends MTBRUpdateable{
	int NumHiddenNeuron, NumInputNeuron;
	double[][] IW,curBeta;
	double[] bias;
	RealMatrix beta, M;
	Random rand = new Random();
	String xmlPath, originalPath, ActivFunc; 
	double delta, hardThreshold;//, testThreshold;
	double[] neuronThresh;
	double[] prevFN, prevFP;

	
	public MultiTargetLRUpdateable(boolean gram, UpdateableClassifier classifier, int MeasureLength, int inputN, int hiddenN, String xmlOriginal, String xmlReduced, String ActivationFunc, double Hthresh, double Tthresh, double delta) {
		super(classifier, MeasureLength);
		NumInputNeuron = inputN;
		NumHiddenNeuron = hiddenN;
		originalPath = xmlOriginal;
		xmlPath = xmlReduced;
		ActivFunc = ActivationFunc;
		hardThreshold = Hthresh;	
		neuronThresh = new double[inputN];
		prevFN = new double[inputN];
		prevFP = new double[inputN];
		for(int i=0; i<neuronThresh.length; i++){
			neuronThresh[i] = Tthresh;
			prevFP[i] = Tthresh;
			prevFN[i] = Tthresh;
		}
		this.delta = delta;
		
		IW = new double[NumHiddenNeuron][NumInputNeuron];
		bias = new double[NumHiddenNeuron];
		
		//make orthogonal hyperplanes!
		gramschmidt GS = new gramschmidt(NumHiddenNeuron, NumInputNeuron+1);
		double[][] hyperWeights;
		if(gram){
			hyperWeights = GS.makeOrthogonals();
		}else{
			hyperWeights = GS.basisVector;
		}
		for(int i=0; i<NumHiddenNeuron; i++){
			for(int j=0; j<NumInputNeuron; j++){
				IW[i][j] = hyperWeights[i][j];
			}
			bias[i] = hyperWeights[i][NumInputNeuron];
		}
	}
	
	/**
	 * Constructor when the input weights are initialized
	 */
	public MultiTargetLRUpdateable(double[][] iw, double[] b, UpdateableClassifier classifier, int MeasureLength, int inputN, int hiddenN, String xmlOriginal, String xmlReduced, String ActivationFunc, double Hthresh, double Tthresh, double delta) {
		super(classifier, MeasureLength);
		NumInputNeuron = inputN;
		NumHiddenNeuron = hiddenN;
		originalPath = xmlOriginal;
		xmlPath = xmlReduced;
		ActivFunc = ActivationFunc;
		hardThreshold = Hthresh;
		neuronThresh = new double[inputN];
		prevFN = new double[inputN];
		prevFP = new double[inputN];
		for(int i=0; i<neuronThresh.length; i++){
			neuronThresh[i] = Tthresh;
			prevFP[i] = Tthresh;
			prevFN[i] = Tthresh;
		}
		this.delta = delta;
		
		this.IW = iw;
		this.bias = b;
	}


	public MultiTargetLRUpdateable(UpdateableClassifier classifier, int size) {
		// TODO Auto-generated constructor stub
		super(classifier,size);
	}


	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
		
		Instances Labels = extractBatchLabels(trainingSet);
		
		double[][] H0;
		
		if(ActivFunc.equals("No")){
			H0 = NoActivationFunction(getWindowMatrix(Labels, true), IW, bias);
		}else if(ActivFunc.equals("HardLim")){
			H0 = hardLimActivationFunction(getWindowMatrix(Labels, true), IW, bias, hardThreshold);
		}else{ //sigmoid function
			H0 = SigmoidActivationFunction(getWindowMatrix(Labels, true), IW, bias);
		}

		Instances reducedData = addNewLabelsToBatchFeatures(trainingSet, H0);
		MultiLabelInstances multiReducedData = new MultiLabelInstances(reducedData, xmlPath);
		numLabels = multiReducedData.getNumLabels();
        labelIndices = multiReducedData.getLabelIndices();
        featureIndices = multiReducedData.getFeatureIndices();
        
        super.buildInternal(multiReducedData);
		
		RealMatrix h0 = MatrixUtils.createRealMatrix(H0);
		M = new SingularValueDecomposition((h0.transpose()).multiply(h0)).getSolver().getInverse();
//		beta = (new SingularValueDecomposition(h0).getSolver().getInverse())
//							.multiply(MatrixUtils.createRealMatrix(getWindowMatrix(Labels, true)));
		beta = M.multiply(h0.transpose().multiply(MatrixUtils.createRealMatrix(getWindowMatrix(Labels, true))));
		curBeta = beta.getData();
		System.out.println(curBeta.length+" "+curBeta[0].length);
	}

	
	public Instances addNewLabelsToBatchFeatures(MultiLabelInstances dataset, double[][] H0) throws Exception{
		//delete original labels from dataset
		Instances newData = new RemoveAllLabels().transformInstances(dataset);
		//add attributes for new labels
		ArrayList<String> values = new ArrayList<String>();
		values.add("0"); values.add("1");
		for(int i=0; i<NumHiddenNeuron; i++)
			newData.insertAttributeAt(new Attribute("hiddenLabel_"+i, values),newData.numAttributes());
				
				
		for(int i=0; i<H0.length; i++){
			for(int j=0; j<H0[i].length; j++){
				newData.instance(i).setValue(newData.attribute("hiddenLabel_"+j), H0[i][j]);
			}
		}
		
		return newData;
	}
	
	public double[][] NoActivationFunction(double[][] P, double[][] IW, double[] bias){
		RealMatrix p = MatrixUtils.createRealMatrix(P);
		RealMatrix iw = MatrixUtils.createRealMatrix(IW);
		RealMatrix V = p.multiply(iw.transpose());
		
		double[][] biasMat = new double[P.length][bias.length];
		for(int i=0; i<P.length; i++)
			biasMat[i] = bias;
		RealMatrix biasMatrix = MatrixUtils.createRealMatrix(biasMat);
		
		V = V.add(biasMatrix);
		double[][] v = V.getData();
		
//		System.out.println("==================================");
//		for(int i=0; i<v.length; i++){
//			for(int j=0; j<v[i].length; j++){
//				System.out.print(v[i][j]+" ");
//			}
//			System.out.println();
//		}
//		System.out.println("==================================");
		
		return v;
	}
	
	public double[][] hardLimActivationFunction(double[][] P, double[][] IW, double[] bias, double thresh){
		RealMatrix p = MatrixUtils.createRealMatrix(P);
		RealMatrix iw = MatrixUtils.createRealMatrix(IW);
		RealMatrix V = p.multiply(iw.transpose());
		
		double[][] biasMat = new double[P.length][bias.length];
		for(int i=0; i<P.length; i++)
			biasMat[i] = bias;
		RealMatrix biasMatrix = MatrixUtils.createRealMatrix(biasMat);
		
		V = V.add(biasMatrix);
		double[][] v = V.getData();
		
		//comment out for runs!
		for(int i=0; i<v.length; i++){
			for(int j=0; j<v[i].length; j++){
				if(v[i][j] >= thresh){ //changed this one 
					v[i][j] = 1;
				}else{
					v[i][j] = 0;
				}
			}
		}
		
		return v;
	}
	
	public double[][] SigmoidActivationFunction(double[][] P, double[][] IW, double[] bias){
		RealMatrix p = MatrixUtils.createRealMatrix(P);
		RealMatrix iw = MatrixUtils.createRealMatrix(IW);
		RealMatrix V = p.multiply(iw.transpose());
		
		double[][] biasMat = new double[P.length][bias.length];
		for(int i=0; i<P.length; i++)
			biasMat[i] = bias;
		RealMatrix biasMatrix = MatrixUtils.createRealMatrix(biasMat);
		
		V = V.add(biasMatrix);
		double[][] v = V.getData();
		for(int i=0; i<v.length; i++){
			for(int j=0; j<v[i].length; j++){
				v[i][j] = 1/(double)(1+Math.exp(-v[i][j]));
			}
		}
		
		return v;
	}
	
	public Instances extractBatchLabels(MultiLabelInstances dataset){
		int[] featIndex = dataset.getFeatureIndices();
		Instances labelInstances = new Instances(dataset.getDataSet());
		for(int i=featIndex.length-1; i>=0; i--)
			labelInstances.deleteAttributeAt(featIndex[i]);
		return labelInstances;
	}
	
	public double[][] getWindowMatrix(Instances batch, boolean labelFlag){
		double[][] labelMatrix = new double[batch.numInstances()][batch.numAttributes()];
		for(int r=0; r<batch.numInstances();r++){
			for(int c=0; c<batch.numAttributes();c++){
				if(labelFlag)
					labelMatrix[r][c] = (int) (2*batch.instance(r).value(c)-1);
				else
					labelMatrix[r][c] = batch.instance(r).value(c);
			}
		}
		return labelMatrix;
	}
	
	

	public void updateClassifierForMLBatch(MultiLabelInstances batch, boolean adaptive, boolean shallowAE) throws Exception {
		Instances Labels = extractBatchLabels(batch);
		double[][] labels = getWindowMatrix(Labels, true); //make labels -1 , +1
		
		//update threshold
		if(adaptive){ //otherwise the values are set in constructor
			int[] FNInLabels = new int[batch.getNumLabels()];
			int[] FPInLabels = new int[batch.getNumLabels()];
			Instances temp = batch.getDataSet();
			double[][] preds = makePredictionForBatch(temp);
			for(int mk=0; mk<temp.numInstances();mk++){
				for(int mj=0; mj<labels[mk].length;mj++){
					if(labels[mk][mj] != preds[mk][mj]){
						if(labels[mk][mj] == 1){
							FNInLabels[mj]++;
						}else{
							FPInLabels[mj]++;
						}
					}
				}
			}
			
			for(int i=0; i<batch.getNumLabels();i++){
				if(FNInLabels[i]<prevFN[i] && FPInLabels[i]<prevFP[i]){
					//no change
				}else if(FNInLabels[i]>prevFN[i] && FPInLabels[i]>prevFP[i]){
//					//increase in false predictions 
					if((FNInLabels[i]-prevFN[i]) > (FPInLabels[i]-prevFP[i])){
						//if error increase in negative side is bigger
						neuronThresh[i] -= delta;
					}else{
						//if error increase in positive side is bigger
						neuronThresh[i] += delta;
					}
				}else if(FNInLabels[i]>prevFN[i] && FPInLabels[i]<prevFP[i]){
					//reduce negative range
					neuronThresh[i] -= delta;
				}else if(FNInLabels[i]<prevFN[i] && FPInLabels[i]>prevFP[i]){
					//reduce positive range
					neuronThresh[i] += delta;
				}
				
				if(neuronThresh[i] < -1)
					neuronThresh[i] = -1;
				if(neuronThresh[i] > 1)
					neuronThresh[i] = 1;
				
				//update FN & FP
				prevFN[i] = FNInLabels[i];
				prevFP[i] = FPInLabels[i];
				
			}
			
			
		}		
		
		
		//update classifiers
		if(shallowAE){
			IW = beta.getData();
		}
		double[][] H;
		if(ActivFunc.equals("No")){
			H = NoActivationFunction(labels, IW, bias);
		}else if(ActivFunc.equals("HardLim")){
			H = hardLimActivationFunction(labels, IW, bias, hardThreshold);
		}else{
			H = SigmoidActivationFunction(labels, IW, bias);
		}
		Instances reducedData = addNewLabelsToBatchFeatures(batch, H);
//		MultiLabelInstances multiReducedData = new MultiLabelInstances(reducedData, xmlPath);
		super.updateClassifierBatch(reducedData);
		
		RealMatrix h = MatrixUtils.createRealMatrix(H);
		// M = M - M * H' * (eye(Block) + H * M * H')^(-1) * H * M
		RealMatrix tmp = (MatrixUtils.createRealIdentityMatrix(batch.getNumInstances())).add(h.multiply(M.multiply(h.transpose())));
		tmp = (new SingularValueDecomposition(tmp).getSolver().getInverse()).multiply(h.multiply(M));
		M = M.subtract(M.multiply(h.transpose().multiply(tmp)));
		
		//beta = beta + M * H' * (Tn - H * beta)
		tmp = MatrixUtils.createRealMatrix(getWindowMatrix(Labels, true)).subtract((h.multiply(beta)));
		beta = beta.add(M.multiply(h.transpose().multiply(tmp)));
		
		curBeta = beta.getData();
		
	}
	

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance inst) {
		Instance instance = inst;
		boolean[] bipartition = new boolean[NumInputNeuron];
		double[] Confidences = new double[NumInputNeuron];
        double[][] internalConfidences = new double[1][numLabels];
        double[][] classLabel = new double[1][numLabels];
               
        //comment for hoeffding
//        RemoveAllLabels.transformInstance(instance, labelIndices);
        
//        instance.insertAttributeAt(/*new Attribute("hiddenLabel_"+counter, values),*/instance.numAttributes());
        
//        ArrayList<String> values = new ArrayList<String>();
//		values.add("0"); values.add("1");

        for (int counter = 0; counter < numLabels; counter++) {
        	//add for hoeffding
        	instance = brt.transformInstance(inst, counter);
            double distribution[];
            try {
                distribution = ensemble[counter].distributionForInstance(instance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            classLabel[0][counter] = distribution[0];

            // The confidence of the label being equal to 1
            internalConfidences[0][counter] = distribution[0];
        }
        
        RealMatrix learnerPred = MatrixUtils.createRealMatrix(classLabel); 
		RealMatrix Ttest = learnerPred.multiply(beta);
		double[][] Tlabels = Ttest.getData();
		//not used yet
		RealMatrix learberConf = MatrixUtils.createRealMatrix(internalConfidences);
		RealMatrix Tconf = learberConf.multiply(beta);
		double[][] Tconfs = Tconf.getData();
		
		for(int i=0; i < NumInputNeuron; i++){
			// Ensure correct predictions both for class values {0,1} and {1,0}
	        bipartition[i] = (Tlabels[0][i]  >= neuronThresh[i]/*== 1*/) ? true : false; /////////////
	        Confidences[i] = Tconfs[0][i];
		}
		
		//confidence is the distribution of class 1 multiplied by beta
        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, Confidences);
        return mlo;
	}

}
