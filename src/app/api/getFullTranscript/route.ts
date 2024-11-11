import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { YoutubeTranscript } from 'youtube-transcript';

export async function GET(request: Request) {
    const user = await auth();

    if (!user) {
        return NextResponse.redirect("/api/auth/login");
    }

    const urlParams = new URL(request.url).searchParams;
    const videoId = urlParams.get("url");
    const lang = urlParams.get("lang") || "en";

    if (!videoId) {
        return NextResponse.json({ error: "Video ID not found in URL" }, { status: 400 });
    }

    try {
        const source = await YoutubeTranscript.fetchTranscript(videoId);
        let transcript=""
        for(const item of source)
        {
            transcript+=item.text+" "
        }
        // const data =await NextResponse.json(transcript).json() as {transcript:string}
        // console.log(data)
        // console.log(data.transcript)
        return NextResponse.json(transcript);
    } catch (error) {
        console.error(error);
        return NextResponse.json({ error: "Failed to fetch transcript" }, { status: 500 });
    }
}
