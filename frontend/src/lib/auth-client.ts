import { createAuthClient } from "./mock-auth";

const authClient = createAuthClient({
  baseURL: "https://ai-native-hackathon-backend.vercel.app",
});

export default authClient;
