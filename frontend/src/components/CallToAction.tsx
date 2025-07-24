"use client"

import { Button } from "@/components/ui/button"
import { ChevronRight, Github, BookOpen } from "lucide-react"
import Link from "next/link"

export function CallToAction() {
    return (
        <section className="bg-primary/5 py-12 md:py-24 mx-auto">
            <div className="container">
                <div className="mx-auto flex max-w-[980px] flex-col items-center gap-4 text-center">
                    <h2 className="text-3xl font-bold leading-tight tracking-tighter md:text-4xl">
                        Ready to add memory to your apps?
                    </h2>
                    <p className="max-w-[750px] text-lg text-muted-foreground">
                        Start building intelligent applications with contextual memory capabilities. 
                        Simple API, powerful results.
                    </p>
                    <div className="flex flex-col gap-4 sm:flex-row mt-6">
                        <Link href="/chat-demo">
                            <Button size="lg">
                                Try the Demo
                                <ChevronRight className="ml-2 h-4 w-4" />
                            </Button>
                        </Link>
                        <Link href="/api-docs">
                            <Button variant="outline" size="lg">
                                <BookOpen className="mr-2 h-4 w-4" />
                                View API Docs
                            </Button>
                        </Link>
                        <Link
                            href="https://github.com/rowantrollope/remember"
                        >
                            <Button variant="outline" size="lg">
                                <Github className="mr-2 h-4 w-4" />
                                GitHub
                            </Button>
                        </Link>
                    </div>
                </div>
            </div>
        </section>
    )
}
