import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import LinearProgress from '@mui/material/LinearProgress';
import Alert from '@mui/material/Alert';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import { ContractService } from '../services/api';

const ContractUpload = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    const validFiles = selectedFiles.filter(file => file.type === 'application/pdf');
    
    if (validFiles.length > 0) {
      setFiles(prevFiles => [...prevFiles, ...validFiles]);
      setError(null);
    } else {
      setError('Please select valid PDF files.');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFiles = Array.from(event.dataTransfer.files);
    const validFiles = droppedFiles.filter(file => file.type === 'application/pdf');
    
    if (validFiles.length > 0) {
      setFiles(prevFiles => [...prevFiles, ...validFiles]);
      setError(null);
    } else {
      setError('Please drop valid PDF files.');
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select at least one file to upload.');
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      // If only one file is selected, navigate to its details page after upload
      if (files.length === 1) {
        const response = await ContractService.uploadContract(files[0], (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        });
        
        // Navigate to the contract details page
        navigate(`/contracts/${response.contract_id}`);
      } else {
        // For multiple files, upload them one by one
        const totalFiles = files.length;
        let completedFiles = 0;
        
        for (const file of files) {
          await ContractService.uploadContract(file, (progressEvent) => {
            // Calculate progress across all files
            const fileProgress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            const overallProgress = Math.round(
              ((completedFiles * 100) + fileProgress) / totalFiles
            );
            setUploadProgress(overallProgress);
          });
          
          completedFiles++;
        }
        
        // Navigate to the dashboard after all files are uploaded
        navigate('/dashboard');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(
        err.response?.data?.detail || 
        'Failed to upload contracts. Please try again.'
      );
      setUploading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h4" component="h1" gutterBottom>
        Upload Contract
      </Typography>
      
      <Paper 
        elevation={3} 
        sx={{ 
          p: 4, 
          mt: 4, 
          borderRadius: 2,
          backgroundColor: 'background.paper' 
        }}
      >
        <Box
          sx={{
            border: '2px dashed',
            borderColor: 'divider',
            borderRadius: 2,
            p: 6,
            textAlign: 'center',
            mb: 3,
            cursor: 'pointer',
            '&:hover': {
              borderColor: 'primary.main',
              backgroundColor: 'rgba(25, 118, 210, 0.04)',
            },
          }}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => document.getElementById('contract-file-input').click()}
        >
          <input
            type="file"
            id="contract-file-input"
            accept="application/pdf"
            style={{ display: 'none' }}
            onChange={handleFileChange}
            multiple
          />
          
          <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
          
          <Typography variant="h6" gutterBottom>
            Drag and drop your contract PDFs here
          </Typography>
          
          <Typography variant="body2" color="text.secondary" gutterBottom>
            or click to browse files (multiple files supported)
          </Typography>
          
          {files.length > 0 && (
            <Box 
              sx={{ 
                mt: 2, 
                p: 1, 
                backgroundColor: 'rgba(25, 118, 210, 0.08)', 
                borderRadius: 1,
                maxHeight: '150px',
                overflowY: 'auto'
              }}
            >
              {files.map((file, index) => (
                <Box 
                  key={index}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    mb: index < files.length - 1 ? 1 : 0,
                    p: 1,
                    borderRadius: 1,
                    backgroundColor: 'rgba(255, 255, 255, 0.5)'
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1, mr: 2 }}>
                    <PictureAsPdfIcon sx={{ mr: 1, color: 'error.main' }} />
                    <Typography variant="body2" noWrap sx={{ maxWidth: '200px' }}>
                      {file.name}
                    </Typography>
                  </Box>
                  <Button 
                    size="small" 
                    color="error" 
                    onClick={() => {
                      setFiles(files.filter((_, i) => i !== index));
                    }}
                  >
                    Remove
                  </Button>
                </Box>
              ))}
            </Box>
          )}
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {uploading && (
          <Box sx={{ width: '100%', mb: 3 }}>
            <LinearProgress variant="determinate" value={uploadProgress} />
            <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 1 }}>
              {uploadProgress}% Uploaded
            </Typography>
          </Box>
        )}

        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Button
            variant="contained"
            size="large"
            onClick={handleUpload}
            disabled={files.length === 0 || uploading}
            startIcon={<CloudUploadIcon />}
          >
            {uploading ? 'Uploading...' : `Upload ${files.length === 1 ? 'Contract' : 'Contracts'} (${files.length})`}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default ContractUpload;
