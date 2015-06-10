#include "svm_template.h"
#include "utils.h"
#include "cuda_runtime_api.h"

///////////////////////////////////////////////////////////////
// Utils

int Utils::StoreResults(char *filename, int *results, unsigned int numResults) {
	unsigned int i;
	FILE *fid;

	if ((fid = fopen(filename, "w")) == NULL) {
		printf("Cannot open file %s\n", filename);
		return -1;
	}
	for (i = 0; i < numResults; i++) {
		fprintf(fid, "%d\n", results[i]); 
	}
	fclose(fid);
	return 0;
}

///////////////////////////////////////////////////////////////
// SvmData

SvmData::SvmData() {
	type = DENSE;
	numClasses = 0;
	numVects = 0;
	dimVects = 0;
	numVects_aligned = 0;
	dimVects_aligned = 0;
	data_raw = NULL;
	class_labels = NULL;
	vector_labels = NULL;
	allocatedByCudaHost = false;
	transposed = false;
	labelsInFloat = false;
	invertLabels = false;
}

SvmData::~SvmData() {
	Delete();
}

int SvmData::Delete() {
	type = DENSE;
	numClasses = 0;
	numVects = 0;
	dimVects = 0;
	numVects_aligned = 0;
	dimVects_aligned = 0;
	if(allocatedByCudaHost) {
		if(data_raw != NULL) cudaFree(data_raw);
		if(class_labels != NULL) cudaFree(class_labels);
		if(vector_labels != NULL) cudaFree(vector_labels);
	} else {
		if(data_raw != NULL) free(data_raw);
		if(class_labels != NULL) free(class_labels);
		if(vector_labels != NULL) free(vector_labels);
	}
	data_raw = NULL;
	class_labels = NULL;
	vector_labels = NULL;
	allocatedByCudaHost = false;
	transposed = false;
	labelsInFloat = false;
	invertLabels = false;
	return SUCCESS;
}

int SvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type, struct svm_memory_dataformat req_data_format) {
	FILE *fid;

	Delete();

	float * labels_tmp = NULL;

	FILE_SAFE_OPEN(fid, filename, "r")

		/* Read data from file. */
		switch(file_type) {
		case LIBSVM_TXT:
			load_libsvm_data_dense(fid, data_type, req_data_format);
			break;
		default:
			printf("Format of the data file not supported or the setting is wrong\n");
			return FAILURE;
	}

	/* Close streams. */
	fclose(fid);

	return SUCCESS;
} //SvmData::Load()

int SvmData::load_libsvm_data_dense(FILE * &fid, SVM_DATA_TYPE data_type, svm_memory_dataformat req_data_format) {

	Delete();
	bool useBuffer = true;

	char *buf,
		*sbuf = NULL,
		*sbuf_tmp,
		*line = NULL,
		*line_end;
	unsigned int i,
		it = 0,
		j,
		start_pos,
		sbuf_size = 64<<20, //intial size 64MB
		read_chars,
		new_len,
		line_len;
	int line_size,
		ret;

	if ((start_pos = ftell(fid)) == EOF) {
		printf("File is not openned!\n");
		exit(EXIT_FAILURE);
	}

	ret = SUCCESS;

	if(req_data_format.labelsInFloat && sizeof(int) != sizeof(float)) REPORT_ERROR("4byte-int platform assumed");

	if (useBuffer) {
		/* Try to store file into buffer. */
		MEM_SAFE_ALLOC(sbuf, char, sbuf_size)
			read_chars = 0;

		while (1) {
			new_len = (unsigned int) fread(sbuf + read_chars, sizeof(char), sbuf_size - read_chars, fid);

			read_chars += new_len;
			if (read_chars == sbuf_size) {
				/* Expanding buffer size. */
				sbuf_size <<= 1;
				sbuf_tmp = (char *) realloc(sbuf, sbuf_size * sizeof(float));

				if (sbuf_tmp == NULL) {
					/* Not enough memory. */
					printf("Warning: Not enough memory - buffering disabled!\n");
					sbuf_size = 0;
					free(sbuf);
					fseek(fid, start_pos, SEEK_SET);
					break;
				} else {
					sbuf = sbuf_tmp;
				}
			} else {
				/* File loaded successfully. */
				sbuf[read_chars++] = 0;
				sbuf_size = read_chars;
				sbuf = (char *) realloc(sbuf, sbuf_size * sizeof(float));
				printf("Buffering input text file (%d B).\n", sbuf_size);
				break;
			}
		}
	}

	if (!useBuffer || sbuf_size == 0) {
		/* Count lines and elements. */
		line = (char *) malloc((line_len = LINE_LENGTH) * sizeof(char));
		while (1) {
			line_size = parse_line(fid, line, line_end, line_len);
			if (line_size < 1) {
				if (line_size == 0) {
					/* Empty line. */
					exit_input_error(numVects + 1);
				} else {
					/* End of file. */
					break;
				}
			}
			/* Skip alpha. */
			buf = next_string_spec_space(line);
			ret |= *buf ^ ' ';
			buf++;


			while (!ret && (buf < line_end) && (*buf != 0)) {
				buf = next_string_spec_colon(buf);
				ret |= *(buf++) ^ ':';
				buf = next_string_spec_space(buf);
				if (*buf == '\n' || *buf == 0) {
					break;
				}
				ret |= *buf ^ ' ';
				buf++;
			}

			if (ret != SUCCESS) {
				exit_input_error(numVects + 1);
			}

			i = strtoi_reverse(last_string_spec_colon(buf));

			if (*buf == '\n') {
				buf++;
			}

			if (i > dimVects) {
				dimVects = i;
			}
			numVects++;
		}
		fseek(fid, start_pos, SEEK_SET);

		numVects_aligned = ALIGN_UP(numVects, req_data_format.vectAlignment);
		dimVects_aligned = ALIGN_UP(dimVects, req_data_format.dimAlignment);
		if (req_data_format.allocate_write_combined) malloc_host_WC((void **) & vector_labels, sizeof(int) * numVects_aligned);
		else {
			if (req_data_format.allocate_pinned) malloc_host_PINNED((void **) & vector_labels, sizeof(int) * numVects_aligned);
			else vector_labels = (int *) malloc(sizeof(float) * numVects_aligned);
		}
		if (req_data_format.allocate_write_combined) malloc_host_WC((void **) & data_raw, sizeof(float) * dimVects_aligned * numVects_aligned);
		else {
			if (req_data_format.allocate_pinned) malloc_host_PINNED((void **) & data_raw, sizeof(float) * dimVects_aligned * numVects_aligned);
			else data_raw = (float *) malloc(sizeof(float) * dimVects_aligned * numVects_aligned);
		}
		memset(data_raw, 0, sizeof(float) * dimVects_aligned * numVects_aligned);

		i = 0;
		while (1) {
			if (parse_line(fid, line, line_end, line_len) == -1) {
				/* End of file. */
				break;
			}
			/* Read alpha. */
			vector_labels[i] = strtol(line, &buf, 10);

			while (buf < line_end) {
				/* Read index. */
				buf = next_string_spec_colon(buf);
				j = strtoi_reverse(buf - 1) - 1;
				buf++;

				/* Read value. */
				if (req_data_format.transposed) data_raw[j * numVects_aligned + i] = strtof_fast(buf, &buf);
				else data_raw[i * dimVects_aligned + j] = strtof_fast(buf, &buf);

				if (*buf == '\n' || *buf == 0) {
					break;
				}
				buf++;
			}
			i++;
		}

		/* Free memory. */
		free(line);
	} else {
		/* Count lines and elements. */
		buf = sbuf;
		while (*buf != 0) {
			/* Skip alpha. */
			buf = next_string_spec_space(buf);
			ret |= *buf ^ ' ';
			buf++;

			while (!ret && (*buf != '\n') && (*buf != 0)) {
				buf = next_string_spec_colon(buf);
				ret |= *(buf++) ^ ':';
				buf = next_string_spec_space(buf);
				if (*buf == '\n' || *buf == 0) {
					break;
				}
				ret |= *buf ^ ' ';
				buf++;
			}

			if (ret != SUCCESS) {
				exit_input_error(numVects + 1);
			}

			i = strtoi_reverse(last_string_spec_colon(buf));

			if (*buf == '\n') {
				buf++;
			}

			if (i > dimVects) {
				dimVects = i;
			}
			numVects++;
		}

		fseek(fid, start_pos, SEEK_SET);

		numVects_aligned = ALIGN_UP(numVects, req_data_format.vectAlignment);
		dimVects_aligned = ALIGN_UP(dimVects, req_data_format.dimAlignment);
		if (req_data_format.allocate_write_combined) malloc_host_WC((void **) & vector_labels, sizeof(float) * numVects_aligned);
		else {
			if (req_data_format.allocate_pinned) malloc_host_PINNED((void **) & vector_labels, sizeof(float) * numVects_aligned);
			else vector_labels = (int *) malloc(sizeof(float) * numVects_aligned);
		}
		if (req_data_format.allocate_write_combined) malloc_host_WC((void **) & data_raw, sizeof(float) * dimVects_aligned * numVects_aligned);
		else {
			if (req_data_format.allocate_pinned) malloc_host_PINNED((void **) & data_raw, sizeof(float) * dimVects_aligned * numVects_aligned);
			else data_raw = (float *) malloc(sizeof(float) * dimVects_aligned * numVects_aligned);
		}
		memset(data_raw, 0, sizeof(float) * dimVects_aligned * numVects_aligned);

		i = 0;
		buf = sbuf;
		while (*buf != 0) {
			/* Read alpha. */
			vector_labels[i] = strtol(buf, &buf, 10);
			buf++;

			while ((*buf != '\n') && (*buf != 0)) {
				/* Read index. */
				buf = next_string_spec_colon(buf);
				j = strtoi_reverse(buf - 1) - 1;
				buf++;

				/* Read value. */
				if (req_data_format.transposed) data_raw[j * numVects_aligned + i] = strtof_fast(buf, &buf);
				else data_raw[i * dimVects_aligned + j] = strtof_fast(buf, &buf);

				if ((*buf == '\n') || (*buf == 0)) {
					break;
				}
				buf++;
			}
			if (*buf == '\n') {
				buf++;
			}
			i++;
		}

		/* Free memory. */
		free(sbuf);
	}

	allocatedByCudaHost = req_data_format.allocate_pinned || req_data_format.allocate_write_combined;
	this->type = DENSE;
	transposed = req_data_format.transposed;

	//make class labels
	int max_idx = -2;
	for(unsigned int i=0; i < numVects; i++) {
		if(max_idx < vector_labels[i]) max_idx = vector_labels[i];
	}
	class_labels = (int *) malloc(sizeof(int) * (max_idx + 2));
	for(int i=0; i < max_idx + 2; i++) class_labels[i] = -2;
	for(unsigned int i=0; i < numVects; i++) {
		int ci = vector_labels[i];
		if(ci < -1) REPORT_ERROR("Class index lower than -1");
		if(class_labels[ci+1] == -2) class_labels[ci+1] = ci;
	}
	numClasses = 0;
	for(int i=0; i < max_idx + 2; i++) {
		if(class_labels[i] != -2) {
			class_labels[numClasses] = class_labels[i];
			numClasses++;
		}
	}

	//store if the first label is negative: to LibSVM compatibility of stored model:
	invertLabels = !(vector_labels[0] == 1);


	//if float labels required:
	if(req_data_format.labelsInFloat) {
		labelsInFloat = true;
		float *p = (float *)vector_labels;
		for(unsigned int i=0; i < numVects; i++) p[i] = (float) vector_labels[i];
	}

	return 0;
} //SvmData::load_libsvm_data_dense



///////////////////////////////////////////////////////////////
// SvmModel
SvmModel::SvmModel() {
	alphas = NULL;
	data = NULL;
	params = NULL;
	allocatedByCudaHost = false;
}


SvmModel::~SvmModel() {
	Delete();
}

int SvmModel::Delete() {
	//don't delete data & params - they are only poiters - they need to be Destroyet themself externally
	data = NULL;
	params = NULL;
	if(allocatedByCudaHost) cudaFree(alphas);
	else free(alphas);
	alphas = NULL;
	return SUCCESS;
}

int SvmModel::StoreModelGeneric(char *model_file_name, SVM_MODEL_FILE_TYPE type) {

	switch(type) {
		case M_LIBSVM_TXT:
			SAFE_CALL(StoreModel_LIBSVM_TXT(model_file_name));
			break;
		default:
			REPORT_ERROR("Unsupported model storage format");
	}

	return SUCCESS;
}

int SvmModel::StoreModel_LIBSVM_TXT(char *model_file_name) {
	//unsigned int i,
	//	j,
	//	width,
	//	height;
	//float alpha,
	//	value,
	//	alpha_mult;
	FILE *fid;

	if (alphas == NULL || data == NULL || params == NULL) {
		return FAILURE;
	}

	FILE_SAFE_OPEN(fid, model_file_name, "w");

	unsigned int height = data->GetNumVects();
	unsigned int width =  data->GetDimVects();

	/* Print header. */
	fprintf(fid, "svm_type c_svc\nkernel_type %s\n", kernel_type_table[params->kernel_type]);
	switch (params->kernel_type) {
	case POLY:
		fprintf(fid, "degree %d\n", params->degree);
	case SIGMOID:
		fprintf(fid, "coef0 %g\n", params->coef0);
	case RBF:
		fprintf(fid, "gamma %g\n", params->gamma);
		break;
	}
	fprintf(fid, "nr_class %d\ntotal_sv %d\n", data->numClasses, params->nsv_class1 + params->nsv_class2);

	//float alpha_mult = (data->class_labels[0] > data->class_labels[1])? -1.0f : 1.0f;
	unsigned int classIds[2] = {0, 1};
	//store labels and counts - to LibSVM compatible
	if(data->invertLabels) {
		fprintf(fid, "rho %g\nlabel %d %d\nnr_sv %d %d\nSV\n", -params->rho, data->class_labels[0], data->class_labels[1], params->nsv_class1, params->nsv_class2);
		classIds[0] = 1;
		classIds[1] = 0;
	} else {
		fprintf(fid, "rho %g\nlabel %d %d\nnr_sv %d %d\nSV\n", params->rho, data->class_labels[1], data->class_labels[0], params->nsv_class2, params->nsv_class1);
	}

	//store positive Support Vectors
	for (unsigned int i = 0; i < height; i++) {
		if(alphas[i] > 0.0f) {
			if(data->labelsInFloat && ((float*)data->vector_labels)[i] != (float) data->class_labels[classIds[1]]) continue;
			if(!data->labelsInFloat && data->vector_labels[i] != data->class_labels[classIds[1]]) continue;
			float a = alphas[i];
			fprintf(fid, "%.16g ", a);

			for (unsigned int j = 0; j < width; j++) {
				float value = data->GetValue(i, j);
				if (value != 0.0F) {
					if (value == 1.0F) {
						fprintf(fid, "%d:1 ", j + 1);
					} else {
						fprintf(fid, "%d:%g ", j + 1, value);
					}
				}
			}

			fprintf(fid, "\n");
		}
	}

	//store negative Support Vectors
	for (unsigned int i = 0; i < height; i++) {
		if(alphas[i] > 0.0f) {
			if(data->labelsInFloat && ((float*)data->vector_labels)[i] != (float) data->class_labels[classIds[0]]) continue;
			if(!data->labelsInFloat && data->vector_labels[i] != data->class_labels[classIds[0]]) continue;
			float a = alphas[i];
			fprintf(fid, "%.16g ", -a);

			for (unsigned int j = 0; j < width; j++) {
				float value = data->GetValue(i, j);
				if (value != 0.0F) {
					if (value == 1.0F) {
						fprintf(fid, "%d:1 ", j + 1);
					} else {
						fprintf(fid, "%d:%g ", j + 1, value);
					}
				}
			}

			fprintf(fid, "\n");
		}
	}


	fclose(fid);
	return SUCCESS;
} //StoreModel

int SvmModel::CalculateSupperVectorCounts() {
	if(alphas == NULL || params == NULL || data == NULL) return FAILURE;

	params->nsv_class1 = 0;
	params->nsv_class2 = 0;
	int *labels = data->GetVectorLabelsPointer();
	int *class_labels = data->GetClassLabelsPointer();
	for(unsigned int i=0; i < data->GetNumVects(); i++) {
		if(alphas[i] > 0) {
			if(data->GetLabelsInFloat()) {
				 if(((float*)labels)[i] == (float) class_labels[0]) params->nsv_class1++;
											                   else params->nsv_class2++;
			} else {
				 if(labels[i] == class_labels[0]) params->nsv_class1++;
										     else params->nsv_class2++;
			}
		}
	}
	return SUCCESS;
}