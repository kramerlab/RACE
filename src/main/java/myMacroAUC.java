
import java.util.ArrayList;

import mulan.evaluation.measure.LabelBasedAUC;
import mulan.evaluation.measure.MacroAverageMeasure;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;

/**
 * Implementation of the macro-averaged AUC measure.
 *
 */
public class myMacroAUC extends LabelBasedAUC implements MacroAverageMeasure {

    /**
     * Creates a new instance of this class
     *
     * @param numOfLabels the number of labels
     */
    public myMacroAUC(int numOfLabels) {
        super(numOfLabels);
    }

    public String getName() {
        return "Macro-averaged AUC";
    }

    public double getValue() {
        double[] labelAUC = new double[numOfLabels];
        ArrayList<Double> availableAUC = new ArrayList<Double>();
        
        for (int i = 0; i < numOfLabels; i++) {
            ThresholdCurve tc = new ThresholdCurve();
            Instances result = tc.getCurve(m_Predictions[i], 1);
            labelAUC[i] = ThresholdCurve.getROCArea(result);
            if( !Double.isNaN(labelAUC[i]) &&  !Double.isInfinite(labelAUC[i])){
            	availableAUC.add(labelAUC[i]);
            }
        }
        
        double mean = 0;
        for(int i=0; i<availableAUC.size();i++)
        	mean += availableAUC.get(i);
        return mean/availableAUC.size();
//        return Utils.mean(labelAUC);
    }

    /**
     * Returns the AUC for a particular label
     * 
     * @param labelIndex the index of the label 
     * @return the AUC for that label
     */
    public double getValue(int labelIndex) {
        ThresholdCurve tc = new ThresholdCurve();
        Instances result = tc.getCurve(m_Predictions[labelIndex], 1);
        return ThresholdCurve.getROCArea(result);  
    }

}