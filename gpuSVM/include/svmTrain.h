#ifndef SVMTRAINH
#define SVMTRAINH

#include "svmCommon.h"

/**
 * Performs SVM training
 * @param data the training points, stored as a flat column major array.
 * @param nPoints the number of training points
 * @param nDimension the dimensionality of the training points
 * @param labels the labels for the training points (+/- 1.0f)
 * @param alpha a pointer to a float buffer that will be allocated by performTraining and contain the unsigned weights of the classifier after training.
 * @param kp a pointer to a struct containing all the information about the kernel parameters.  The b offset from the training process will be stored in this struct after training is complete
 * @param cost the training cost parameter C
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param transposedData the training points, stored as a flat row major array.  This pointer can be omitted.
 */
void performTraining(float* data, int nPoints, int nDimension, float* labels, float** alpha, Kernel_params* kp, float cost, SelectionHeuristic heuristicMethod = ADAPTIVE, float epsilon = 1e-5f, float tolerance = 1e-3f, float* transposedData = 0);

/**
 * Densifies model by collecting Support Vectors from training set.
 * @param trainingPoints the training points, stored as a flat column major array.
 * @param nTrainingPoints the number of training points
 * @param nDimension the dimensionality of the training points
 * @param trainingAlpha the weights learned during the training process
 * @param trainingLabels the labels of the training points (+/- 1.0f)
 * @param p_supportVectors a pointer to a float array, where the Support Vectors will be stored as a flat column major array
 * @param p_nSV a pointer to the number of Support Vectors in this model
 * @param p_alpha a pointer to a float array, where the Support Vector weights will be stored
 * @param epsilon an optional parameter controling the threshold for which points are considered Support Vectors.  Default is 1e-5f.
 */

void formModel(float* trainingPoints, int nTrainingPoints, int nDimension, float* trainingAlpha, float* trainingLabels, float** p_supportVectors, int* p_nSV, float** p_alpha, float epsilon = 1e-5f);



#endif
