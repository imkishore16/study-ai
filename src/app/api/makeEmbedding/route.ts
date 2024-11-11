import { env } from "@/env";
import { auth } from "@/lib/auth";
import Concept2 from "next-auth/providers/concept2";

export const runtime = "edge";

export async function POST(req: Request): Promise<Response> {

    const user = await auth();

    if (!user?.user?.email) {
      return new Response("Saved locally | Login for Cloud Sync", {
        status: 401,
      });
    }

    const body = await req.json() as { transcript: string };
    const { transcript } = body;
    // console.log(body)
    console.log(transcript)

    try {
        const saveEmbedding = await fetch(`${env.BACKEND_BASE_URL}/api/v1/add`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `${env.AUTH_SECRET}`
          },
          body: JSON.stringify({
            source: transcript,
            user: user.user.email,
            note_id: new Date().getTime(),
          }),
        });
    
        if (saveEmbedding.status !== 200) {
          console.error("Failed to save embedding");
        }
      } catch (error) {
        console.error("Error occurred while saving embedding: ", error);
      }

    return new Response(transcript);
}
