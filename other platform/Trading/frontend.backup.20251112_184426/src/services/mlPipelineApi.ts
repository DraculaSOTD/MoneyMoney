import axios from './axiosConfig';
import type { 
  DataPipelineJob, 
  ModelCatalog, 
  DataFetchConfig, 
  PreprocessConfig 
} from '../features/mlModels/mlModelsSlice';

export interface DataFetchRequest extends DataFetchConfig {
  profile_id: number;
}

export interface PreprocessRequest extends PreprocessConfig {
  profile_id: number;
  data_id: string;
}

export interface TrainingRequest {
  profile_id: number;
  model_id: number;
  data_id: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  validation_split: number;
}

class MLPipelineApi {
  // Data fetching
  async fetchData(request: DataFetchRequest): Promise<DataPipelineJob> {
    const response = await axios.post('/api/ml/data/fetch', request);
    return response.data;
  }

  async preprocessData(request: PreprocessRequest): Promise<DataPipelineJob> {
    const response = await axios.post('/api/ml/data/preprocess', request);
    return response.data;
  }

  async getJobStatus(jobId: string): Promise<DataPipelineJob> {
    const response = await axios.get(`/api/ml/training/status/${jobId}`);
    return response.data;
  }

  async cancelJob(jobId: string): Promise<void> {
    await axios.delete(`/api/ml/jobs/${jobId}`);
  }

  // Model catalog
  async getModelCatalog(): Promise<ModelCatalog[]> {
    const response = await axios.get('/api/ml/models/catalog');
    return response.data;
  }

  // Training
  async startTraining(request: TrainingRequest): Promise<DataPipelineJob> {
    const response = await axios.post('/api/ml/training/start', request);
    return response.data;
  }

  // Job polling utility
  async pollJobStatus(
    jobId: string, 
    onUpdate: (job: DataPipelineJob) => void,
    interval = 1000
  ): Promise<DataPipelineJob> {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const job = await this.getJobStatus(jobId);
          onUpdate(job);
          
          if (job.status === 'completed') {
            resolve(job);
          } else if (job.status === 'failed') {
            reject(new Error(job.error || 'Job failed'));
          } else if (job.status !== 'cancelled') {
            setTimeout(poll, interval);
          }
        } catch (error) {
          reject(error);
        }
      };
      
      poll();
    });
  }
}

export default new MLPipelineApi();