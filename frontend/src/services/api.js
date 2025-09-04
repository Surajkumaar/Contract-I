import axios from 'axios';

// Create axios instance with default config
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Contract API services
const ContractService = {
  // Get all contracts
  getAllContracts: async () => {
    try {
      const response = await api.get('/contracts');
      return response.data;
    } catch (error) {
      console.error('Error fetching contracts:', error);
      throw error;
    }
  },

  // Get contract status
  getContractStatus: async (contractId) => {
    try {
      const response = await api.get(`/contracts/${contractId}/status`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching status for contract ${contractId}:`, error);
      throw error;
    }
  },

  // Get contract data
  getContractData: async (contractId) => {
    try {
      const response = await api.get(`/contracts/${contractId}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching data for contract ${contractId}:`, error);
      throw error;
    }
  },

  // Upload contract
  uploadContract: async (file, onUploadProgress) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/contracts/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress,
      });

      return response.data;
    } catch (error) {
      console.error('Error uploading contract:', error);
      throw error;
    }
  },
};

export { ContractService };
export default api;
