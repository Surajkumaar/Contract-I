import axios from 'axios';

// API base URL for direct links (downloads, etc.)
// In Docker environment, use the service name instead of localhost
export const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Disable SSL verification warnings for local development
axios.defaults.validateStatus = function () {
  return true;
};

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
  
  // Get download URL for contract PDF
  getContractDownloadUrl: (contractId) => {
    return `${API_BASE_URL}/contracts/${contractId}/download`;
  },
};

export { ContractService };
export default api;
