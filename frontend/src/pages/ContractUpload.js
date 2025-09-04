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
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError(null);
    } else {
      setFile(null);
      setError('Please select a valid PDF file.');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'application/pdf') {
      setFile(droppedFile);
      setError(null);
    } else {
      setError('Please drop a valid PDF file.');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to upload.');
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      const response = await ContractService.uploadContract(file, (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        setUploadProgress(percentCompleted);
      });

      // Navigate to the contract details page
      navigate(`/contracts/${response.contract_id}`);
    } catch (err) {
      console.error('Upload error:', err);
      setError(
        err.response?.data?.detail || 
        'Failed to upload contract. Please try again.'
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
          />
          
          <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
          
          <Typography variant="h6" gutterBottom>
            Drag and drop your contract PDF here
          </Typography>
          
          <Typography variant="body2" color="text.secondary" gutterBottom>
            or click to browse files
          </Typography>
          
          {file && (
            <Box 
              sx={{ 
                mt: 2, 
                p: 1, 
                backgroundColor: 'rgba(25, 118, 210, 0.08)', 
                borderRadius: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <PictureAsPdfIcon sx={{ mr: 1, color: 'error.main' }} />
              <Typography variant="body2" noWrap>
                {file.name}
              </Typography>
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
            disabled={!file || uploading}
            startIcon={<CloudUploadIcon />}
          >
            {uploading ? 'Uploading...' : 'Upload Contract'}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default ContractUpload;
