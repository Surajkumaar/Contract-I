import React, { useState, useEffect } from 'react';
import { Link as RouterLink } from 'react-router-dom';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import RefreshIcon from '@mui/icons-material/Refresh';
import ContractCard from '../components/ContractCard';
import { ContractService } from '../services/api';

const Dashboard = () => {
  const [contracts, setContracts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchContracts = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await ContractService.getAllContracts();
      setContracts(data);
    } catch (err) {
      console.error('Error fetching contracts:', err);
      setError('Failed to load contracts. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchContracts();
    
    // Set up polling for in-progress contracts
    const intervalId = setInterval(() => {
      const hasProcessingContracts = contracts.some(
        contract => contract.status === 'processing' || contract.status === 'uploaded'
      );
      
      if (hasProcessingContracts) {
        fetchContracts();
      }
    }, 5000);
    
    return () => clearInterval(intervalId);
  }, [contracts]);

  const handleRefresh = () => {
    fetchContracts();
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" component="h1">
          Contract Dashboard
        </Typography>
        <Box>
          <Button 
            variant="outlined" 
            startIcon={<RefreshIcon />} 
            onClick={handleRefresh}
            sx={{ mr: 2 }}
          >
            Refresh
          </Button>
          <Button 
            variant="contained" 
            component={RouterLink} 
            to="/upload" 
            startIcon={<UploadFileIcon />}
          >
            Upload Contract
          </Button>
        </Box>
      </Box>

      {loading && contracts.length === 0 ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 8 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      ) : contracts.length === 0 ? (
        <Box sx={{ textAlign: 'center', mt: 8 }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No contracts uploaded yet
          </Typography>
          <Button 
            variant="contained" 
            component={RouterLink} 
            to="/upload" 
            startIcon={<UploadFileIcon />}
            sx={{ mt: 2 }}
          >
            Upload Your First Contract
          </Button>
        </Box>
      ) : (
        <Grid container spacing={3}>
          {contracts.map((contract) => (
            <Grid item key={contract._id} xs={12} sm={6} md={4}>
              <ContractCard contract={contract} />
            </Grid>
          ))}
        </Grid>
      )}
    </Container>
  );
};

export default Dashboard;
