import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Divider from '@mui/material/Divider';
import Chip from '@mui/material/Chip';
import LinearProgress from '@mui/material/LinearProgress';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import RefreshIcon from '@mui/icons-material/Refresh';
import { ContractService } from '../services/api';

const ContractView = () => {
  const { contractId } = useParams();
  const navigate = useNavigate();
  const [contract, setContract] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchContractData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // First get the status
      const statusData = await ContractService.getContractStatus(contractId);
      setStatus(statusData);
      
      // Get the contract data for any status
      try {
        const contractData = await ContractService.getContractData(contractId);
        setContract(contractData);
      } catch (dataErr) {
        // If contract data isn't available yet, just continue with status
        console.log('Contract data not available yet:', dataErr);
        // Don't set error here, as we still have status information
      }
    } catch (err) {
      console.error('Error fetching contract data:', err);
      setError('Failed to load contract data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Use ref to track processing status without causing effect to re-run
  const processingStatusRef = React.useRef(false);
  
  // Update ref when status changes
  useEffect(() => {
    if (status) {
      processingStatusRef.current = status.status === 'processing' || status.status === 'uploaded';
    }
  }, [status]);

  useEffect(() => {
    fetchContractData();
    
    // Poll for updates if the contract is still processing with longer interval
    const intervalId = setInterval(() => {
      if (processingStatusRef.current) {
        fetchContractData();
      }
    }, 10000); // Increased to 10 seconds to reduce server load
    
    return () => clearInterval(intervalId);
  }, [contractId]); // Only depend on contractId, not status

  const handleRefresh = () => {
    fetchContractData();
  };

  const handleBack = () => {
    navigate('/');
  };

  const getStatusColor = (statusValue) => {
    switch (statusValue) {
      case 'uploaded':
        return 'info';
      case 'processing':
        return 'warning';
      case 'completed':
        return 'success';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const renderContractSection = (title, data) => {
    if (!data) return null;
    
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          {typeof data === 'object' ? (
            <Grid container spacing={2}>
              {Object.entries(data).map(([key, value]) => (
                <Grid item xs={12} sm={6} key={key}>
                  <Typography variant="subtitle2" color="text.secondary">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {value ? (typeof value === 'object' ? JSON.stringify(value, null, 2) : value) : 'N/A'}
                  </Typography>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Typography variant="body1">{data}</Typography>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderAdditionalFields = (additionalFields) => {
    if (!additionalFields) return null;
    
    return (
      <Card sx={{ mb: 3, borderLeft: '4px solid', borderColor: 'secondary.main' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Additional Fields
            <Chip 
              label="Dynamic" 
              color="secondary" 
              size="small" 
              sx={{ ml: 1, fontSize: '0.7rem' }} 
            />
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          {typeof additionalFields === 'object' ? (
            <Grid container spacing={2}>
              {Object.entries(additionalFields).map(([key, value]) => (
                <Grid item xs={12} sm={6} key={key}>
                  <Typography variant="subtitle2" color="text.secondary">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {value ? (typeof value === 'object' ? JSON.stringify(value, null, 2) : value) : 'N/A'}
                  </Typography>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Typography variant="body1">{additionalFields}</Typography>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Button 
          startIcon={<ArrowBackIcon />} 
          onClick={handleBack}
        >
          Back to Dashboard
        </Button>
        
        <Button 
          startIcon={<RefreshIcon />} 
          onClick={handleRefresh}
        >
          Refresh
        </Button>
      </Box>

      <Typography variant="h4" component="h1" gutterBottom>
        Contract Details
      </Typography>

      {loading && !status ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 8 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      ) : status ? (
        <Box>
          <Paper sx={{ p: 3, mb: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ mr: 2 }}>
                Status:
              </Typography>
              <Chip 
                label={status.status} 
                color={getStatusColor(status.status)} 
                sx={{ textTransform: 'capitalize' }}
              />
            </Box>
            
            {status.status === 'processing' && (
              <Box sx={{ width: '100%', mt: 2 }}>
                <LinearProgress variant="determinate" value={status.progress || 0} />
                <Typography variant="body2" color="text.secondary" align="right" sx={{ mt: 0.5 }}>
                  {status.progress || 0}% Complete
                </Typography>
              </Box>
            )}
            
            {status.error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {status.error}
              </Alert>
            )}
          </Paper>

          {contract ? (
            <Box>
              <Paper sx={{ p: 3, mt: 4 }}>
                <Typography variant="h6" gutterBottom>
                  Extracted Contract Data (Raw JSON)
                </Typography>
                <Box 
                  sx={{ 
                    backgroundColor: 'grey.100', 
                    p: 2, 
                    borderRadius: 1,
                    overflowX: 'auto',
                    maxHeight: '70vh',
                    overflow: 'auto'
                  }}
                >
                  <pre style={{ margin: 0 }}>
                    {JSON.stringify(contract.extracted_data || contract, null, 2)}
                  </pre>
                </Box>
              </Paper>
            </Box>
          ) : status.status === 'error' ? (
            <Alert severity="error" sx={{ mt: 2 }}>
              {status.error || 'An error occurred during contract processing.'}
            </Alert>
          ) : (
            <Box sx={{ textAlign: 'center', mt: 8 }}>
              <CircularProgress />
              <Typography variant="h6" sx={{ mt: 2 }}>
                {status.status === 'processing' ? 'Processing contract...' : 'Waiting for processing to begin...'}
              </Typography>
            </Box>
          )}
        </Box>
      ) : (
        <Alert severity="error" sx={{ mt: 2 }}>
          Contract not found
        </Alert>
      )}
    </Container>
  );
};

export default ContractView;
