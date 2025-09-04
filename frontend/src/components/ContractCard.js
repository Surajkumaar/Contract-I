import React from 'react';
import { useNavigate } from 'react-router-dom';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardActions from '@mui/material/CardActions';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import LinearProgress from '@mui/material/LinearProgress';
import VisibilityIcon from '@mui/icons-material/Visibility';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import Chip from '@mui/material/Chip';

const ContractCard = ({ contract }) => {
  const navigate = useNavigate();
  
  const getStatusColor = (status) => {
    switch (status) {
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

  const handleViewClick = () => {
    navigate(`/contracts/${contract._id}`);
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flexGrow: 1 }}>
        <Typography variant="h6" component="div" noWrap title={contract.filename}>
          {contract.filename}
        </Typography>
        
        <Box sx={{ mt: 2, mb: 1 }}>
          <Chip 
            label={contract.status} 
            color={getStatusColor(contract.status)} 
            size="small" 
            sx={{ textTransform: 'capitalize' }}
          />
        </Box>
        
        {contract.status === 'processing' && (
          <Box sx={{ width: '100%', mt: 2 }}>
            <LinearProgress variant="determinate" value={contract.progress || 0} />
            <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 0.5 }}>
              {contract.progress || 0}%
            </Typography>
          </Box>
        )}
        
        {contract.error && (
          <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', color: 'error.main' }}>
            <ErrorOutlineIcon fontSize="small" sx={{ mr: 1 }} />
            <Typography variant="body2" color="error">
              {contract.error}
            </Typography>
          </Box>
        )}
      </CardContent>
      
      <CardActions>
        <Button 
          size="small" 
          startIcon={<VisibilityIcon />} 
          onClick={handleViewClick}
          disabled={contract.status === 'uploaded' || contract.status === 'processing'}
        >
          View Details
        </Button>
      </CardActions>
    </Card>
  );
};

export default ContractCard;
