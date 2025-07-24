"use client"

import { Button } from "@/components/ui/button"
import { ChevronRight, Github } from "lucide-react"
import Link from "next/link"

export function HeroSection() {
    return (
        <section className="container py-6 md:py-12 lg:py-14 mx-auto">
            <div className="mx-auto flex max-w-[980px] flex-col items-center gap-4 text-center">
                <h1 className="text-3xl font-bold text-center leading-tight tracking-tighter md:text-5xl lg:text-6xl lg:leading-[1.1]">
                    Save and Retrieve{" "}
                    <span className="text-red-500">Memories</span> with
                    Redis Remem
                    <br />
                    Powered by Redis VectorSet
                </h1>
                <p className="max-w-[750px] text-lg text-muted-foreground md:text-xl">
                    Seamlessly integrate AI-powered memory capabilities into
                    your applications. Fast, intelligent, and easy to use for
                    storing and retrieving contextual information.
                </p>
                <p className="max-w-[750px] text-lg md:text-xl">
                    
                </p>
                <div className="flex flex-col gap-4 sm:flex-row">
                    <Link href="/chat-demo">
                        <Button size="lg">
                            Try Demo
                            <ChevronRight className="ml-2 h-4 w-4" />
                        </Button>
                    </Link>
                    <Link
                        href="https://github.com/rowantrollope/remember"
                    >
                        <Button variant="outline" size="lg">
                            <Github className="mr-2 h-4 w-4" />
                            View on GitHub
                        </Button>
                    </Link>
                </div>
            </div>
        </section>
    )
}
