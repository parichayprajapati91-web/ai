
import { z } from 'zod';
import { voiceDetectionRequestSchema, voiceDetectionResponseSchema, insertApiKeySchema, apiKeys, requestLogs } from './schema';

export const api = {
  voiceDetection: {
    detect: {
      method: 'POST' as const,
      path: '/api/voice-detection',
      input: voiceDetectionRequestSchema,
      responses: {
        200: voiceDetectionResponseSchema,
        400: z.object({ status: z.literal("error"), message: z.string() }),
        401: z.object({ status: z.literal("error"), message: z.string() }),
      },
    },
    forensics: {
      method: 'POST' as const,
      path: '/api/voice-forensics',
      input: z.object({
        audioBase64: z.string(),
        language: z.string().optional(),
        gender: z.enum(['male', 'female', 'unknown']).optional(),
      }),
      responses: {
        200: z.object({
          status: z.literal("success"),
          classification: z.enum(['AI', 'HUMAN', 'UNCERTAIN']),
          human_probability: z.number(),
          ai_probability: z.number(),
          confidence: z.number(),
          forensic_analysis: z.record(z.any()),
        }),
        400: z.object({ status: z.literal("error"), message: z.string() }),
        401: z.object({ status: z.literal("error"), message: z.string() }),
      },
    },
  },
  admin: {
    generateKey: {
      method: 'POST' as const,
      path: '/api/admin/generate-key',
      input: z.object({ owner: z.string() }),
      responses: {
        201: z.custom<typeof apiKeys.$inferSelect>(),
      },
    },
    getStats: {
      method: 'GET' as const,
      path: '/api/admin/stats',
      responses: {
        200: z.object({
          totalRequests: z.number(),
          aiDetected: z.number(),
          humanDetected: z.number(),
          recentLogs: z.array(z.custom<typeof requestLogs.$inferSelect>()),
        }),
      },
    },
    deleteLogs: {
      method: 'DELETE' as const,
      path: '/api/admin/delete-logs',
      input: z.object({ logIds: z.array(z.number()) }),
      responses: {
        200: z.object({ status: z.literal("success"), message: z.string() }),
        400: z.object({ status: z.literal("error"), message: z.string() }),
      },
    }
  }
};
