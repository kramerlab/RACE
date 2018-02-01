import java.util.ArrayList;
import java.util.List;

import weka.classifiers.UpdateableClassifier;
import weka.core.Instances;
import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.Measure;


public interface myMultiLabelLearnerInterface extends UpdateableClassifier{
	
	public void build(MultiLabelInstances trainingSet) throws Exception;
	
	public void keepAllMeasures(List<Measure> m);
	

	public void updateClassifierBatch(Instances instances)throws Exception;
	
	public ArrayList<Double>[] measureGetter();
	
	public double[][] makePredictionForBatch(Instances instances)throws InvalidDataException, ModelInitializationException, Exception;

}
