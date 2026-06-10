import axios from 'axios';

const client = axios.create({ timeout: 30000 });

export default client;
